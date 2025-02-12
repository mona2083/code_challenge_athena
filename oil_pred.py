import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.api import add_constant

from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
from scipy import stats
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import grangercausalitytests
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

from sklearn.metrics import mean_squared_error as MSE 
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")
import japanize_matplotlib

# アウトプットディレクトリの設定
OUTPUT_DIR = "output/"
# ディレクトリが存在しない場合、ディレクトリを作成する
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

target_col = "OT"
TEST_DATE = "2018-06-01"
OUTLIER_DICT = {}
PRED_DICT = {}


def plot_outlier(df, col, path, ewm_span=720, threshold=3.0):
    """
    移動平均の算出と外れ値の計測
    Parameters
    ----------
    df : pd.DataFrame
        対象のデータフレーム
    col : string
        対象カラム名
    path : string
        画像保存先Path
    ewm_span : int
        移動平均の期間
    threshold : float
        閾値の分散の倍数
    Returns
    -------
    pd.DataFrame
        外れ値のデータ
    """
    
    dd = df.copy()
    ds = dd[col]
    fig, ax = plt.subplots(figsize=(16,4))
    # 移動平均の計算
    ewm_mean = ds.ewm(span=ewm_span).mean()
    # 分散の計算
    ewm_std = ds.ewm(span=ewm_span).std()
    
    # データフレームの作成
    dd['mean'] = ewm_mean
    dd['std'] = ewm_std
    dd['thresh_max'] = dd['mean'] + dd['std']*threshold
    dd['thresh_min'] = dd['mean'] - dd['std']*threshold
    dd['fixed'] = np.nan

    # 外れ値と認定された行に代入する値を決定
    for idx in dd.index:
        if dd.loc[idx, col] > dd.loc[idx, 'thresh_max']:
            dd.loc[idx, 'fixed'] = dd.loc[idx, 'thresh_max']
        elif dd.loc[idx, col] < dd.loc[idx, 'thresh_min']:
            dd.loc[idx, 'fixed'] = dd.loc[idx, 'thresh_min']
    # 外れ値と認定された行のみに絞る
    outlier = dd.loc[dd['fixed'].notnull(), [col,'mean','std','fixed']]
    
    # 図の描写
    ax.plot(ds, label='original')
    ax.plot(ewm_mean, label='ewma')
    ax.fill_between(ds.index,
                    ewm_mean - ewm_std * threshold,
                    ewm_mean + ewm_std * threshold,
                    alpha=0.2)
    ax.scatter(outlier.index, outlier[col], label='outlier')
    ax.legend(['実数値','移動平均値','外れ値'])
    plt.title(f"移動平均値と外れ値: {col}")
    plt.savefig(f"{path}")

    return outlier

def data_decomposition(ds, model, title, path, period=24):
    """
    decompositonの実行
    Parameters
    ----------
    ds : pd.Seriese
        対象のSeriese
    model : string
        {"additive", "multiplicative"} decompositionのモデル
    title : string
        画像タイトル
    path : string
        画像保存先Path
    period : int
        decompositionのperiod
    """
    # decompositionの実行
    dcmp = seasonal_decompose(ds, model=model, period=period)
    # Plotの描写
    plt.rc("figure",figsize=(20,8))
    fig = dcmp.plot().suptitle(title)
    plt.tight_layout()
    plt.savef

def data_ac_pac(ds, path, lag=50):
    """
    AC, PACの計算とPlot
    Parameters
    ----------
    ds : pd.Seriese
        対象のSeriese
    path : string
        画像保存先Path
    lag : int
        acfにおけるnlagsの指定
    """
    # ACの計算
    df_acf = sm.tsa.stattools.acf(ds, nlags=lag)
    # PACの計算
    df_pacf = sm.tsa.stattools.pacf(ds, method='ols', nlags=lag)

    # Plotの描写
    plt.rc("figure",figsize=(20,4))
    fig = sm.graphics.tsa.plot_acf(ds, lags=lag)
    plt.savefig(f"{path}_ac.png")
    fig = sm.graphics.tsa.plot_pacf(ds, lags=lag)
    plt.savefig(f"{path}_pac.png")

def propeht_fit(df, date, add_cols, title, img_path, add_seasonality=None, model_path=None):
    """
    Prophetのモデル
    Parameters
    ----------
    df : pd.DataFrame
        対象のデータフレーム
    date : string
        テストデータ、学習データに分割する日付
    add_cols : list
        説明変数として追加するカラム名のリスト
    title : string
        画像タイトル
    img_path : string
        画像保存先Path or None
    ewm_span : int
        移動平均の期間
    add_seasonality : list
        季節性を手動で追加する際のリスト: ['daily','weekly','monthly','yearly'] or None
    model_path : string
        モデルの保存先 or None
    Returns
    -------
    pd.DataFrame
        予測値のデータフレーム
    """
    # データを学習データ、テストデータに分ける
    data_pr = df.copy().reset_index(drop=False).rename(columns={"date":"ds","OT":"y"})
    data_pr['cap'] = 3
    data_pr_train = data_pr.loc[data_pr['ds']<date]
    data_pr_test = data_pr.loc[data_pr['ds']>=date]

    # Prophetの定義
    # 季節性手動追加が指定されている場合はここではそれぞれOffにする
    if add_seasonality is not None:
        pf = Prophet(
            weekly_seasonality=False,
            yearly_seasonality=False,
        )
    else:
        pf = Prophet()
    # 説明変数の追加   
    for col in add_cols:
        pf.add_regressor(col)
    # 手動で季節性を追加する場合
    if add_seasonality is not None:
        if 'daily' in add_seasonality:
            pf.add_seasonality(name='daily', period=1, fourier_order=2)
        if 'weekly' in add_seasonality:
            pf.add_seasonality(name='weekly', period=7, fourier_order=5, mode='multiplicative', prior_scale=3)
        if 'monthly' in add_seasonality:
            pf.add_seasonality(name='monthly', period=30.5, fourier_order=3)
        if 'yearly' in add_seasonality:
            pf.add_seasonality(name='yearly', period=365, fourier_order=3)
    # モデルのFitting
    pf.fit(data_pr_train)

    # 予測値の算出
    forecast = pf.predict(data_pr_test)
    forecast_df = forecast.merge(data_pr_test,on='ds').set_index('ds')
    
    rmse = np.sqrt(MSE(forecast_df['y'], forecast_df['yhat']))   #　RMSEの計算
    mae = MAE(forecast_df['y'], forecast_df['yhat'])  #　MAEの計算
    print(f"{title[:3]} 使用した説明変数: {add_cols}")
    print(f"{title[:3]} RMSE : % f" %(rmse)) 
    print(f"{title[:3]} MAE : % f" %(mae)) 

    # 画像保存先Pathが指定されている場合に画像の描写と保存
    if img_path is not None:
        plt.figure(figsize=(18,4))
        forecast_df['y'].plot()
        forecast_df['yhat'].plot()
        plt.legend(['実数値','予測値'])
        plt.xlabel('')
        plt.title(f"{title}: RMSE={rmse:.3f}, MAE={mae:.3f}")        
        plt.savefig(img_path)

    # モデル保存先Pathが指定されている場合にモデルの保存
    if model_path is not None:
        with open(model_path, 'w') as fout:
            fout.write(model_to_json(pf))  # Save model
        
    return forecast_df

def main()
    # データの読み込み
    data = pd.read_csv('ett.csv')
    # 基本情報のPrint
    print(data.info())

    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')

    # check if there is missing time or date
    whole_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='h')
    if len(whole_index) == data.shape[0]:
        print(f"{data.index.min()}と{data.index.max()}の間に空白期間はありません")
    else:
        print(f"{data.index.min()}と{data.index.max()}の間に空白期間があります")


    print(data.describe())


    # 01_各カラムのPlot
    var_nums = len(data.columns)
    col_num = 1
    raw_num = (var_nums // col_num) if (var_nums % col_num) == 0 else (var_nums // col_num) + 1

    fig = plt.figure(figsize=(20,raw_num*4))

    for i in range(var_nums):
        ax = fig.add_subplot(raw_num,col_num,i+1)
        sns.lineplot(data=data, x=data.index, y=data.columns[i], ax=ax)

        plt.title(f"Var: {data.columns[i]}")
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}01_全カラム図.png")


    # 02_目的変数と説明変数との相関図
    tar_col = target_col
    plot_cols = [ col for col in data.columns if col != tar_col ]
    var_nums = len(plot_cols)
    col_num = 2
    raw_num = (var_nums // col_num) if (var_nums % col_num) == 0 else (var_nums // col_num) + 1

    for i in range(var_nums):
        sns.jointplot(data=data, x=plot_cols[i], y=tar_col,  kind="reg", scatter_kws={'s': 2,'alpha':0.5})
        plt.savefig(f"{OUTPUT_DIR}02_目的変数と説明変数との相関図_{var_nums}.png")


    # 03_目的変数と説明変数との相関図2
    sns.pairplot(data=data)
    plt.savefig(f"{OUTPUT_DIR}03_目的変数と説明変数との相関図2.png")


    # 04_目的変数と説明変数との相関値
    plt.figure(figsize=(10, 6))
    mask = np.triu(np.ones_like(data.corr(), dtype='bool'))
    heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, cmap='coolwarm',
                        annot=True, 
                        mask=mask, 
                        annot_kws={"size": 8}
                        )
    plt.savefig(f"{OUTPUT_DIR}04_目的変数と説明変数との相関値.png")


    # 05_移動平均値と外れ値（目的変数）
    var = target_col
    path = f"{OUTPUT_DIR}05_移動平均値と外れ値_{var}.png"
    outlier = plot_outlier(data, var, path)
    OUTLIER_DICT = {target_col:outlier}


    # 06_decomposition（目的関数）_hourly_2018-01
    data_2018 = data.loc[(data.index>='2018-01-01')&(data.index<'2018-02-01')]
    var = target_col
    model = "additive"
    title = f"Hourly: 2018-01 {model} Decomposition"
    path = f"{OUTPUT_DIR}06_decomposition_OT_hourly_2018-01.png"
    data_decomposition(data_2018[var], model, title, path, period=24)


    # 07_AC&PAC（目的変数）_hourly_2018-01
    lag = 50
    data2plot = data_2018[target_col]
    path = f"{OUTPUT_DIR}07_AC&PAC_{target_col}_hourly_2018-01"
    data_ac_pac(data2plot,path,lag)


    # 08_decomposition（目的変数）_daily
    data_grpd_day = data.copy()
    data_grpd_day['day'] = data_grpd_day.index.date
    data_grpd_day = data_grpd_day.groupby('day').mean()
    var = target_col
    model = "additive"
    title = f"Daily: {model} Decomposition"
    path = f"{OUTPUT_DIR}08_decomposition_OT_daily.png"
    data_decomposition(data_grpd_day[var], model, title, path, period=30)


    # 09_AC&PAC（目的変数）_daily
    lag = 60
    data2plot = data_grpd_day[target_col]
    path = f"{OUTPUT_DIR}09_AC&PAC_{target_col}_saily"
    data_ac_pac(data2plot,path,lag)


    # 10 Granger Causality test
    for col in ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL',]:
        print(f"{col}--------------------------")
        grangercausalitytests(data[['OT', col]], maxlag=2)
        print(f"-----------------------------")
        print("")


    # 11_AC&PAC（目的変数）_daily
    for col in ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL',]:
        var = col
        path = f"{OUTPUT_DIR}11_移動平均値と外れ値_{var}.png"
        ot = plot_outlier(data, col, path)
        OUTLIER_DICT[col] = ot


    # 12_decomposition（説明変数）_daily
    for var in ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL',]:
        data_2018
        model = "additive"
        title = f"Hourly: {model} Decomposition"
        path = f"{OUTPUT_DIR}12_decomposition_{var}_hourly.png"
        data


    # 13_各カラムの外れ値を平均+-分散*3の値に修正
    data_fixed = data.copy()
    for col in data_fixed.columns:
        out_df = OUTLIER_DICT[col]
        for idx in out_df.index:
            data_fixed.loc[idx, col] = out_df.loc[idx,'fixed']


    # 14_1回目_AIモデル予測と実数値の比較
    exp_vars = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    data4pr = data_fixed
    add_cols = exp_vars
    seasonality = None
    title = f"1回目_AIモデル予測と実数値の比較"
    path = f"{OUTPUT_DIR}14_{title}.png"
    pred = propeht_fit(data4pr, TEST_DATE, add_cols, title, path, seasonality)
    PRED_DICT[1] =  pred


    out_std = float(OUTLIER_DICT['OT'].loc[OUTLIER_DICT['OT'].index>='2018-06-01', 'std'].mean())
    print(f"2018年6月以降の移動平均の分散平均： {out_std:.3f}")

    # 15_2回目_AIモデル予測と実数値の比較_seasonalityの追加
    data4pr = data_fixed
    add_cols = exp_vars
    seasonality = ['daily','weekly','monthly','yearly']
    title = f"2回目_AIモデル予測と実数値の比較"
    path = f"{OUTPUT_DIR}15_{title}.png"
    pred = propeht_fit(data4pr, TEST_DATE, add_cols, title, path, seasonality)
    PRED_DICT[2] =  pred


    # 16_3回目_AIモデル予測と実数値の比較_seasonalityの追加
    data4pr = data_fixed.loc[data_fixed.index>='2018-01-10']
    add_cols = exp_vars
    seasonality = ['daily','weekly']
    title = f"3回目_AIモデル予測と実数値の比較"
    path = f"{OUTPUT_DIR}16_{title}.png"
    save_model_path = f"{OUTPUT_DIR}16_model.json"
    pred = propeht_fit(data4pr, TEST_DATE, add_cols, title, path, seasonality, save_model_path)
    PRED_DICT[3] =  pred


    # 17_3回目_モデルから説明変数を一つずつ削除して試す
    for col in exp_vars:
        data4pr = data_fixed.loc[data_fixed.index>='2018-01-10']
        add_cols = [ var for var in exp_vars if var != col ]
        seasonality = ['daily','weekly']
        title = "try"
        path = None
        _ = propeht_fit(data4pr, TEST_DATE, add_cols, title, path, seasonality)

if __name__ == "__main__":
    main()