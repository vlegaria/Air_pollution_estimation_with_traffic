# This file is part of the article:
# "Leveraging Remote Traffic Data for Hyperlocal Air Pollutant Estimation:
# A Scenario-Based Machine Learning Study Across London Monitoring Sites"
#
# Copyright (C) 2026 The authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
import matplotlib as mpl
import os
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.stats.stattools import medcouple
from pathlib import Path

def adjusted_boxplot_bounds(x):
    x = x.dropna().values    
    Q1 = np.percentile(x, 25)
    Q3 = np.percentile(x, 75)
    IQR = Q3 - Q1
    MC = medcouple(x)    
    if MC >= 0:
        lower = Q1 - 1.5 * np.exp(-4 * MC) * IQR
        upper = Q3 + 1.5 * np.exp(3 * MC) * IQR
    else:
        lower = Q1 - 1.5 * np.exp(-3 * MC) * IQR
        upper = Q3 + 1.5 * np.exp(4 * MC) * IQR
        
    return lower, upper, MC

def clean_non_negative(df, cols, pollutant=None):
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df[cols] = df[cols].where(df[cols] >= 0, np.nan)
    if pollutant:
        df = df.dropna(subset=[pollutant])
    return df


dir_drive = Path(r"Datasets_to_train")
filepath2save = Path("Figs")
os.makedirs(filepath2save, exist_ok=True)
feature_fontsize = 26
tick_fontsize = 13
corr_fontsize = 30
cbar_fontsize = 30
stations_considered = ['Camden Kerbside Urban Traffic','London Marylebone Road Urban Traffic','Westminster - Oxford Street Urban Traffic', 'Camden - Euston Road Urban Traffic','Wandsworth - Putney High Street Urban Traffic']
stationname_type = {'CamdenKerbside':'Camden Kerbside Urban Traffic','LondonMaryleboneRoad':'London Marylebone Road Urban Traffic','Westminster-OxfordStreet':'Westminster - Oxford Street Urban Traffic', 'Camden-EustonRoad':'Camden - Euston Road Urban Traffic','Wandsworth-PutneyHighStreet':'Wandsworth - Putney High Street Urban Traffic'}
files =  ["LondonMaryleboneRoad",'CamdenKerbside',"Wandsworth-PutneyHighStreet", 'Camden-EustonRoad', 'Westminster-OxfordStreet']
files_to_check = os.listdir(dir_drive)
other_variables = ['NO', 'NOx as NO2', 'SO2', 'CO']

for station in files:
    print(station)
    df_case = pd.read_csv(os.path.join(dir_drive, f"{station}.csv"))
    df_case = pd.read_csv(os.path.join(dir_drive, f"{station}.csv"))
    df_case["date"] = pd.to_datetime(df_case["date"])
    df_case["wind_deg_ow"] = (np.degrees(np.arctan2(df_case["wdr_sin"], df_case["wdr_cos"])) % 360).round(2)
    df_case["weekday"] = df_case["date"].dt.dayofweek
    df_case["month"] = df_case["date"].dt.month
    df_case["hour"] = df_case["date"].dt.hour
    df_case.drop(columns=["hour_cos", "hour_sin", 'wdr_sin',"wdr_cos", "month_sin", "month_cos", "weekday_sin", "weekday_cos"], inplace=True)
    columns_station = [col for col in df_case.columns if ("bkg" not in col) and ("trf" not in col) and col not in other_variables]
    df_case = df_case[columns_station]
    df_plot = df_case.select_dtypes(include=[np.number]).copy()
    new_cmap = sns.diverging_palette(195, 105, s=90, l=55, as_cmap=True)
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    n_vars = len(df_plot.columns)
    fig, axes = plt.subplots(
        n_vars,
        n_vars,
        figsize=(2.8 * n_vars, 2.8 * n_vars)
    )
    if n_vars == 1:
        axes = np.array([[axes]])
    for i, y_var in enumerate(df_plot.columns):
        for j, x_var in enumerate(df_plot.columns):
            ax = axes[i, j]
            x = df_plot[x_var]
            y = df_plot[y_var]
            valid = x.notna() & y.notna()
            x = x[valid]
            y = y[valid]
            if i == j:
                # Diagonal: histogram
                ax.hist( x, bins=20, color=[0.177423, 0.437527, 0.557565], edgecolor="black")

            elif i > j:
                # Lower triangle: pairwise scatter
                ax.scatter(
                    x,
                    y,
                    s=8,
                    alpha=0.20,
                    color=[0.262138, 0.242286, 0.520837]
                )
            else:
                # Upper triangle: Spearman correlation coefficient
                r = x.corr(y, method="spearman")
                ax.set_facecolor(new_cmap(norm(r)))
                ax.text(
                    0.5,
                    0.5,
                    f"{r:.2f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=corr_fontsize,
                    fontweight="bold",
                    color="black"
                )

                # Hide axes only in upper triangle
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(
                    bottom=False,
                    left=False,
                    labelbottom=False,
                    labelleft=False
                )
            # Show numeric ticks in all cells
            ax.tick_params(
                axis="both",
                which="both",
                labelsize=tick_fontsize,
                labelbottom=True,
                labelleft=True,
                bottom=True,
                left=True
            )
            # Rotate numeric tick labels
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
                tick.set_fontsize(tick_fontsize)

            for tick in ax.get_yticklabels():
                tick.set_rotation(0)
                tick.set_fontsize(tick_fontsize)
            # Feature names:
            # X labels only on the bottom row, vertical
            if i == n_vars - 1:
                ax.set_xlabel(x_var, fontsize=feature_fontsize, rotation=90, labelpad=12)
            else:
                ax.set_xlabel("")
            # Y labels only on the first column, horizontal
            if j == 0:
                ax.set_ylabel(y_var, fontsize=feature_fontsize, rotation=0, labelpad=95)
                ax.yaxis.set_label_coords(-1.15, 0.5)
            else:
                ax.set_ylabel("")

    sm = mpl.cm.ScalarMappable(cmap=new_cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.18, 0.025, 0.77])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Spearman correlation", fontsize=cbar_fontsize)
    cbar.ax.tick_params(labelsize=cbar_fontsize)
    fig.subplots_adjust(
        left=0.16,
        right=0.90,
        bottom=0.18,
        top=0.95,
        wspace=0.25,
        hspace=0.25
    )
    plt.savefig(os.path.join(filepath2save,f"Correlation_Matrix_{station}.png"), dpi=300, bbox_inches="tight", facecolor="white" )
    plt.close()
    