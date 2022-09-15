import mysql.connector
import sqlalchemy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from os.path import exists
import itertools
import matplotlib.pyplot as plt
from scipy import stats
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error
from io import StringIO
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
import math
import matplotlib.pyplot as plt
import json
from IPython.display import display, HTML
plt.rcParams.update({"font.size": 7})
plt.rcParams['image.cmap'] = "Accent"
from matplotlib import cm
cmap = cm.get_cmap('Accent')
import matplotlib as mpl
mpl.rcParams['image.cmap'] = 'Accent'
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from scipy.special import comb


# mysql -h 94.130.228.49 -u apyreader -pSimpleReader42!
con = mysql.connector.connect(user='apyreader', password='SimpleReader42!',
                              host='94.130.228.49', database='apy',
                              auth_plugin='mysql_native_password')
dbcon = sqlalchemy.create_engine('mysql+mysqlconnector://apyreader:SimpleReader42!@94.130.228.49/apy')


def plot_apy_tvl(df, dfp, corr_pools):
    cmap = mpl.cm.get_cmap('Paired')

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
    axs = axs.flatten()
    for i, pool in enumerate(df[df.index.isin(corr_pools)].index.values):
        dfl = df[df.index == pool]
        project = dfl[["chain", "project", "symbol"]].agg(", ".join, axis=1).values[0] + " = " + str(
            dfl["corr14"].values[0].round(3))
        dpv = dfp.loc[pool]
        axs[i].set_title(project)
        ax2 = axs[i].twinx()

        dpv.reset_index().plot(x="dt", y=f"apy", label="apy", color=cmap(4), ax=axs[i], secondary_y=False, grid=True) # color="#ffaaaa",
        dpv.reset_index().plot(x="dt", y=f"dblexp", kind="scatter", color=cmap(11), label="dblexp", ax=axs[i], marker="+", s=10, secondary_y=False, grid=True) #  color="#808080",
        dpv.reset_index().plot(x="dt", y=f"tvl", label="tvl", ax=ax2, color=cmap(0), secondary_y=True, grid=True) # color="#aaaaff",
    fig.tight_layout()



def corr_distribution(df):
    cmap = mpl.cm.get_cmap('Paired')
    fig = plt.figure(figsize=(8,5))
    ax1 = fig.add_subplot(121)
    df[df["corr14"].notna()]["corr14"].hist(ax=ax1, bins=20, grid=True, color=cmap(0))
    df[df["corr14"].notna()]["corr14"].plot.kde(ax=ax1, secondary_y=True, title="kde", grid=True, color=cmap(7))
    ax2 = fig.add_subplot(122)
    bbox=[0, 0, 1, 1]
    ax2.axis('off')
    df2 = df[df["corr14"].notna()][["corr14"]].describe().round(2)
    df2["corr14"] = df2["corr14"].map('{:.2f}'.format)
    mpl_table = ax2.table(cellText = df2.values, rowLabels = df2.index, colLabels=df2.columns, bbox=bbox)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(10)
    fig.tight_layout()

def apy_prediction_charts(df, dfp, corr_pools):
    cmap = mpl.cm.get_cmap('Paired')
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
    axs = axs.flatten()
    for i, pool in enumerate(df[df.index.isin(corr_pools)].index.values):
        dfl = df[df.index == pool]
        project = dfl[["chain", "project", "symbol"]].agg(", ".join, axis=1).values[0] + " = " + str(
            dfl["corr14"].values[0].round(3))
        d = json.loads(dfl["w_json"].values[0])
        dpv = dfp.loc[(pool, slice(None)), :].copy()
        dpv.sort_values(by="tvl", ascending=True, inplace=True)
        # Это наши фичи, по которым считаем модельное значение и еще "единички" - intercept
        cols = ["tvl", "1/tvl", "log.tvl", "tvl2", "1/tvl2", "1/log.tvl"]
        dpv["1/tvl"] = 1 / dpv["tvl"]
        dpv["log.tvl"] = np.log(dpv["tvl"])
        dpv["tvl2"] = dpv["tvl"] * dpv["tvl"]
        dpv["1/tvl2"] = 1 / dpv["tvl2"]
        dpv["1/log.tvl"] = 1 / dpv["log.tvl"]
        # Восстанавливаем предсказание по коэффициентам линейной регрессии
        dpv["pred"] = d["intercept"] + sum([d[c] * dpv[c] for c in cols])
        dpv_today = dpv.loc[(pool, ["2022-09-13"]), :]
        axs[i].set_title(project)
        dpv.plot(kind="scatter", x="tvl", y=f"apy", label="apy", color=cmap(0), ax=axs[i], secondary_y=False, grid=True)
        dpv.plot(kind="line", x="tvl", y=f"pred", label="pred", color=cmap(5), ax=axs[i], secondary_y=False, grid=True)
        dpv_today.plot(kind="scatter", x="tvl", y=f"apy", label="apy-today", color=cmap(3), ax=axs[i], secondary_y=False,
                       marker="^", s=70, grid=True)
        dpv_today.plot(kind="scatter", x="tvl", y=f"pred", label="pred-today", color=cmap(5), ax=axs[i],
                       secondary_y=False, marker="*", s=100, grid=True)
    fig.tight_layout()




def calc_hacker_risk(df, dfs):
    df["hacker_risk"] = 0
    t = len(df.index)
    b = 10

    for i, row in dfs.iterrows():
        project = i[0]
        p = row["cnt"]
        df.loc[df.project==project, "hacker_risk"] = (1 - comb(int(t-p), b) / comb(t, b))
    return df



def calc_regression(df, dfp):
    FROM = 28
    TO = 14
    df1 = df[df.cnt > FROM].copy()
    df1["apy-train.real"] = 0
    df1["apy-test.real"] = 0
    df1["apy-train.pred"] = 0
    df1["apy-test.pred"] = 0
    df1["apy.pred"] = 0
    df1["apy.mae"] = 0
    df1["mae.train"] = 999999
    df1["mae.test"] = 999999
    df1["min.train"] = 999999
    df1["min.test"] = 999999

    pools = [c for c in df1.index.tolist()]
    for pool in pools:
        cond = df1.index == pool
        dfi = dfp.loc[pool].iloc[-FROM:].copy()  #.loc  [dfp.index[-FROM:], [col, des]]
        d = json.loads(df1.loc[cond, "w_json"].values[0])
        # Это наши фичи, по которым считаем модельное значение и еще "единички" - intercept
        cols = ["tvl", "1/tvl", "log.tvl", "tvl2", "1/tvl2", "1/log.tvl"]
        dfi["1/tvl"] = 1 / dfi["tvl"]
        dfi["log.tvl"] = np.log(dfi["tvl"])
        dfi["tvl2"] = dfi["tvl"] * dfi["tvl"]
        dfi["1/tvl2"] = 1 / dfi["tvl2"]
        dfi["1/log.tvl"] = 1 / dfi["log.tvl"]
        # Восстанавливаем предсказание по коэффициентам линейной регрессии
        dfi["pred"] = d["intercept"] + sum([d[c] * dfi[c] for c in cols])
        # df1.loc[cond, "apy-min"] = dfi[col].min()
        curr = df1.loc[cond, :]
        df1.loc[cond, "min.train"] = dfi.loc[dfi.index[-FROM:-TO], "apy"].min().round(5)
        df1.loc[cond, "apy-train.real"] = dfi.loc[dfi.index[-FROM:-TO], "apy"].mean().round(5)
        df1.loc[cond, "apy-train.pred"] = np.abs(dfi.loc[dfi.index[-FROM:-TO], "pred"].mean().round(5))

        df1.loc[cond, "min.test"] = dfi.loc[dfi.index[-TO:], "apy"].min().round(5)
        df1.loc[cond, "apy-test.real"] = dfi.loc[dfi.index[-TO:], "apy"].mean().round(5)
        df1.loc[cond, "apy-test.pred"] = np.minimum(np.abs(dfi.loc[dfi.index[-TO:], "pred"].mean().round(5)),
                                                    df1.loc[cond, "apy-train.real"])
        df1.loc[cond, "mae.train"] = mean_absolute_error(df1.loc[cond, "apy-train.real"], df1.loc[cond, "apy-train.pred"])
        df1.loc[cond, "mae.test"] = mean_absolute_error(df1.loc[cond, "apy-test.real"], df1.loc[cond, "apy-test.pred"])
        df1.loc[cond, "apy.pred"] = np.abs(
            d["intercept"] + curr["tvl"] * d["tvl"] + (1 / curr["tvl"]) * d["1/tvl"] + (np.log(curr["tvl"])) * d[
                "log.tvl"] + (curr["tvl"] * curr["tvl"]) * d["tvl2"] + (1 / (curr["tvl"] * curr["tvl"])) * d["1/tvl2"] + (
                    1 / np.log(curr["tvl"])) * d["1/log.tvl"])

    df1["apy.mae"] = np.abs(df1["apy"] - df1["apy.pred"])
    return df1


def simulate(df, amount, label):
    dfi = df.copy()
    dfi.sort_values(by=label, ascending=False, inplace=True)
    dfi["amount"] = 0
    dfi["emp.risk.win"] = 0
    # dfi["win"] = 0
    # # dfi["E(win)"] = 0
    # dfi["asset"] = 0
    # dfi["der"] = 0
    #
    # dfi["win"] = df[label] * dfi["amount"] / 365 / 100 * 14
    # dfi["asset"] = dfi["win"] * dfi["apyBase"] / dfi["apy"]
    # dfi["der"] = dfi["win"] * dfi["apyReward"] / dfi["apy"]
    # dfi["asset"] = dfi["asset"].fillna(0)
    # dfi["der"] = dfi["der"].fillna(0)
    # dfi["win"] = dfi["asset"] + dfi["der"]
    #
    # dfi.sort_values(by="win", ascending=False, inplace=True)

    selected_projects = []
    skipped_idx = []
    cumsum = 0
    n = 0
    am = amount
    for i, row in dfi.iterrows():
        part = min(row["tvl"] * 0.1, am)
        dfi.loc[i, "amount"] = part
        dfi.loc[i, "win"] = dfi.loc[i, label] * dfi.loc[i, "amount"] / 365 / 100 * 14
        # dfi.loc[i, "E(win)"] = part_risk * (-part) + (1-part_risk) * dfi.loc[i, "win"]

        if selected_projects.count(row.project) > 2:
            skipped_idx.append(i)
            continue

        selected_projects.append(row.project)

        cumsum += part
        am -= part
        n += 1
        if cumsum >= amount:
            break

    # dfi["win"] = df[label] * dfi["amount"] / 365 / 100 * 14
    dfi["asset"] = dfi["win"] * dfi["apyBase"] / dfi["apy"]
    dfi["der"] = dfi["win"] * dfi["apyReward"] / dfi["apy"]
    dfi["asset"] = dfi["asset"].fillna(0)
    dfi["der"] = dfi["der"].fillna(0)
    dfi["win"] = dfi["asset"] + dfi["der"]
    dfi["emp.risk.win"] = dfi["trust"] * (1 - dfi["hacker_risk"]) * (dfi["amount"] + dfi["win"]) + (1 - dfi["trust"] * (1 - dfi["hacker_risk"]))*(-dfi["amount"])


    indexer = ~dfi.index.isin(skipped_idx)
    dfii = dfi.loc[indexer].iloc[:n]
    return dfii
