
# This file is part of the article:
# "Leveraging Remote Traffic Data for Local Air Pollutant Estimation:
# A Scenario-Based Machine Learning Study Across London Monitoring Sites"
#
# Copyright (C) 2026 The authors
##
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


import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Tuple
from pathlib import Path

summary_file = Path("preprocessing_data_counts.csv")
df_preprocessing_counts = pd.read_csv(summary_file)
pollutant_stations = {
    "Camden - Euston Road": ["NO2"],
    "Camden Kerbside": ["NO2", "PM10", "PM25"],
    "London Marylebone Road": ["NO2", "PM10", "PM25", "O3"],
    "Wandsworth - Putney High Street": ["NO2", "PM10"],
    "Westminster - Oxford Street": ["NO2"]
}

def initial_missing_run(series: pd.Series) -> int:
    """
    Returns the number of consecutive missing values at the beginning
    of a series.
    """
    is_missing = series.isna().to_numpy()

    if len(is_missing) == 0:
        return 0

    non_missing_positions = np.flatnonzero(~is_missing)

    if len(non_missing_positions) == 0:
        return len(series)

    return int(non_missing_positions[0])

def clean_non_negative(df, cols):
    df = df.dropna(axis=1, how="all")
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df[cols] = df[cols].where(df[cols] >= 0, np.nan)
    return df


def find_nearest_by_type(dist_df, stations, suffix, k=2):
    """
    Find k nearest stations ending with a given suffix
    for each station in stations list.
    """
    results = {}
    candidates = [s for s in dist_df.index if s.endswith(suffix) and s not in stations]
    for station in stations:
        valid_candidates = [c for c in candidates if c != station]   
        distances = dist_df.loc[station, valid_candidates]
        nearest = distances.sort_values().head(k)        
        results[station] = nearest    
    return results


def make_outer_folds(df_case: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    if "date" in df_case.columns:
        df_case = df_case.sort_values("date").reset_index(drop=True)

    X_dummy = np.arange(len(df_case))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    return list(tscv.split(X_dummy))


def make_inner_folds(X_train: pd.DataFrame, n_splits: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    X_dummy = np.arange(len(X_train))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    return list(tscv.split(X_dummy))


def first_inner_train_size(
    df: pd.DataFrame,
    n_outer_splits: int = 5,
    n_inner_splits: int = 3,
) -> int:
    outer_folds = make_outer_folds(
        df,
        n_splits=n_outer_splits
    )

    first_outer_train_idx, _ = outer_folds[0]

    first_outer_train = df.iloc[
        first_outer_train_idx
    ].copy()

    inner_folds = make_inner_folds(
        first_outer_train,
        n_splits=n_inner_splits
    )

    first_inner_train_idx, _ = inner_folds[0]

    return len(first_inner_train_idx)

def audit_against_first_inner_train(
    df: pd.DataFrame,
    datetime_col: str = "date",
    n_outer_splits: int = 5,
    n_inner_splits: int = 3,
) -> pd.DataFrame:

    data = (
        df
        .sort_values(datetime_col)
        .reset_index(drop=True)
        .copy()
    )

    first_train_size = first_inner_train_size(
        data,
        n_outer_splits=n_outer_splits,
        n_inner_splits=n_inner_splits,
    )

    rows = []

    for column in data.columns:
        if column in [datetime_col, "row_id_original"]:
            continue

        initial_missing = initial_missing_run(
            data[column]
        )

        rows.append({
            "variable": column,
            "initial_missing_count": initial_missing,
            "initial_missing_percentage": round(
                100 * initial_missing / len(data),
                2
            ),
            "first_inner_train_size": first_train_size,
            "empty_in_first_inner_train": (
                initial_missing >= first_train_size
            ),
        })

    return pd.DataFrame(rows)

traffic_vars = ["traffic_level", "currenttraveltime", ]

dist_df = pd.read_csv(r"distance_matrix_km_btw_stations.csv")
dist_df.set_index("Sitename_type", inplace=True)
dir_aux_stations = r"Bkg_or_with_traffic\Neighbouring_stations"
dir_weather = r"Bkg_or_with_traffic\Open_weather"
dir_drive = r"Bkg_or_with_traffic\target_stations_with_traffic"

files = ["London Marylebone Road 23jun2025-12dec2025_withtraffic.csv","Wandsworth - Putney High Street 23jun2025-12dec2025_withtraffic.csv", "Camden - Euston Road 23jun2025-12dec2025_withtraffic.csv", "Westminster - Oxford Street 23jun2025-12dec2025_withtraffic.csv", "Camden Kerbside 23jun2025-12dec2025_withtraffic.csv"]
stations_considered = ["Camden Kerbside Urban Traffic","London Marylebone Road Urban Traffic","Westminster - Oxford Street Urban Traffic", "Camden - Euston Road Urban Traffic","Wandsworth - Putney High Street Urban Traffic"]
stationname_type = {"CamdenKerbside":"Camden Kerbside Urban Traffic","LondonMaryleboneRoad":"London Marylebone Road Urban Traffic","Westminster-OxfordStreet":"Westminster - Oxford Street Urban Traffic", "Camden-EustonRoad":"Camden - Euston Road Urban Traffic","Wandsworth-PutneyHighStreet":"Wandsworth - Putney High Street Urban Traffic"}
files_to_check = os.listdir(dir_aux_stations)
pollutants = [ "NO2", "O3", "PM10","PM25"]

dependant_variables = ["O3","NO", "NO2", "NOx as NO2", "SO2", "CO", "PM10", "PM25"]
stations_map =  { "Camden-EustonRoad": "Camden - Euston Road",
    "CamdenKerbside": "Camden Kerbside",
    "LondonMaryleboneRoad": "London Marylebone Road",
    "Wandsworth-PutneyHighStreet": "Wandsworth - Putney High Street",
    "Westminster-OxfordStreet": "Westminster - Oxford Street"
}
filepath2save = "Datasets_to_train_pollutants"
start_date = pd.Timestamp("2025-06-23", tz="UTC")
end_date = pd.Timestamp("2025-12-13", tz="UTC")

new_counts = []
os.makedirs(filepath2save, exist_ok=True)
for n_station,file in enumerate(files):
    match = re.match(r"^[A-Za-z\s-]+", file)
    if match:
        result = match.group()
        station = result.replace(" ", "")
    print(station)
    #"""
    df = pd.read_csv(os.path.join(dir_drive, file))
    df["date"] = pd.to_datetime(df["date"], errors="raise",utc=True)
    
    df["weekday"] = df["date"].dt.dayofweek
    df["month"] =df["date"].dt.month
    
    weather_file = "OpenWeather_"+station+".csv"
    dir_weather_file = os.path.join(dir_weather,weather_file)
    df_weather = pd.read_csv(dir_weather_file)
    df_weather["date"] = pd.to_datetime( df_weather["date"], errors="raise", utc=True )
    
    df_weather = df_weather[(df_weather["date"] >= start_date) & (df_weather["date"] < end_date)].copy()
    n_openweather_raw = len(df_weather)
    
    df = df.merge(df_weather, on="date", how="inner")
    
    df["wdr_sin"] = np.sin(np.deg2rad(df["wind_deg_ow"])).round(4)
    df["wdr_cos"] = np.cos(np.deg2rad(df["wind_deg_ow"])).round(4)

    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12).round(4)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12).round(4)

    df["weekday_sin"] = np.sin(2 * np.pi * (df["weekday"] - 1) / 7).round(4)
    df["weekday_cos"] = np.cos(2 * np.pi * (df["weekday"] - 1) / 7).round(4)

    cols_to_drop = [c for c in df.columns if c in ["year_month", "hour", "month", "weekday", "currentspeed", "freeflowspeed", "freeflowtraveltime", "traveltime_level","roadclosure", "confidence", "WND", "WSP", "TMP","feels_like_ow", "wind_deg_ow" ]]
    df = df.drop(columns=cols_to_drop)

    # Takes the 2 nearest background and 2 nearest traffic stations and merge them with the main df
    nearest_background = find_nearest_by_type(dist_df,stations_considered, suffix="Background", k=2)
    nearest_traffic = find_nearest_by_type(dist_df, stations_considered, suffix="Traffic", k=2 )

    names_bkg = nearest_background[stationname_type[station]].index.tolist()
    clean_names_bkg = [name.split("Urban")[0].strip() for name in names_bkg]
    #print(station, clean_names_bkg)
    for n_bks, back_station in enumerate(clean_names_bkg):
        for file_ in files_to_check:
            if back_station in file_:
                bkg_df = pd.read_csv(os.path.join(dir_aux_stations, file_))
                bkg_df["date"] = pd.to_datetime( bkg_df["date"], errors="raise", utc=True)
                
                bkg_df = bkg_df[(bkg_df["date"] >= start_date) & (bkg_df["date"] < end_date)].copy()
                pollutants_in = [col for col in bkg_df.columns if col in dependant_variables]
                pollutants_in.insert(0, "date")
                bkg_df = bkg_df[pollutants_in]
                bkg_df = bkg_df.rename(columns={col: f"{col}_bkg{n_bks}"  for col in bkg_df.columns if col != "date" })
                if n_bks == 0:
                    n_bkg0_raw = len(bkg_df)
                elif n_bks == 1:
                    n_bkg1_raw = len(bkg_df)
                    
                df = df.merge(bkg_df, on="date", how="inner")
                
    names_traffic = nearest_traffic[stationname_type[station]].index.tolist()
    clean_names_trf = [name.split("Urban")[0].strip() for name in names_traffic]
    #print(station, clean_names_trf)
    for n_trf, trf_station in enumerate(clean_names_trf):
        for file_ in files_to_check:
            if trf_station in file_:
                trf_df = pd.read_csv(os.path.join(dir_aux_stations, file_))
                trf_df["date"] = pd.to_datetime(trf_df["date"], errors="raise", utc=True)
                
                trf_df = trf_df[(trf_df["date"] >= start_date) & (trf_df["date"] < end_date)].copy()
                pollutants_in = [col for col in trf_df.columns if col in dependant_variables]
                pollutants_in.insert(0, "date")
                trf_df = trf_df[pollutants_in]
                trf_df = trf_df.rename(columns={col: f"{col}_trf{n_trf}"  for col in trf_df.columns if col != "date" })
                if n_trf ==0:
                    n_trf0_raw = len(trf_df)
                    
                elif n_trf == 1:
                    n_trf1_raw = len(trf_df)
                                    
                df = df.merge(trf_df, on="date", how="inner")

    n_after_all_sources_merge = len(df)
    numeric_cols = df.columns.drop(["date", "hour_cos", "hour_sin", "month_cos", "month_sin", "weekday_sin", "weekday_cos", "wdr_sin", "wdr_cos"])
    df = clean_non_negative(df, numeric_cols)


    
    audit_target = audit_against_first_inner_train(
        df,
        datetime_col="date",
        n_outer_splits=5,
        n_inner_splits=3,
    )

    problematic = audit_target[
        audit_target["empty_in_first_inner_train"]
    ]

    if not problematic.empty:
        print(
            f"\nStation={station}"
        )
        print(
            problematic[
                [
                    "variable",
                    "initial_missing_count",
                    "initial_missing_percentage",
                    "first_inner_train_size",
                ]
            ].to_string(index=False)
        )
    problematic_variables = audit_target.loc[audit_target["empty_in_first_inner_train"],"variable"].tolist()

    print("Variables removed:", problematic_variables)
    df = df.drop(columns=problematic_variables, errors="ignore")

    n_after_cleanning = len(df)
    station_name = stations_map[station]
    for pollutant in pollutant_stations[station_name]:
        new_counts.append({"station": station_name, "pollutant": pollutant, "n_openweather_raw": n_openweather_raw, "n_bkg0_raw": n_bkg0_raw, "n_bkg1_raw": n_bkg1_raw, "n_trf0_raw": n_trf0_raw, 
                           "n_trf1_raw": n_trf1_raw, "n_after_all_sources_merge": n_after_all_sources_merge, "n_after_cleanning": n_after_cleanning})
    
    df.to_csv(os.path.join(filepath2save, f"{station}.csv"), index=False)
    print(trf_df["date"].min())
    print(trf_df["date"].max())
    print(len(trf_df))
    #"""
df_new_counts = pd.DataFrame(new_counts)
df_preprocessing_counts = df_preprocessing_counts.merge(df_new_counts, on=["station", "pollutant"], how="left")
df_preprocessing_counts.to_csv(summary_file, index=False)
