# This file is part of the article:
# "Leveraging Remote Traffic Data for Local Air Pollutant Estimation:
# A Scenario-Based Machine Learning Study Across London Monitoring Sites"
#
# Copyright (C) 2026 The authors
#
# Authors are affiliated with Queen Mary University of London
# and Instituto Politécnico Nacional.
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

from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from pyproj import Transformer


N_OUTER_SPLITS = 5
RANDOM_STATE = 754
OUTPUT_DIR = Path(r"interpolation_results_IDW")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TRAFFIC_VARS = [
    "currentspeed",
    "freeflowspeed",
    "traffic_level",
    "currenttraveltime",
    "freeflowtraveltime",
    "traveltime_level",
]

TEMPORAL_VARS = ["hour", "weekday", "month"]

MET_VARS = [
    "temp_ow",
    "pressure_ow",
    "humidity_ow",
    "clouds_ow",
    "wind_speed_ow",
    "wind_deg_ow",
    "dew_point_ow",
]

def find_nearest_by_type(dist_df, stations, suffix, k=2):
    # Find k nearest stations ending with a given suffix  for each station in stations list.
    results = {}
    candidates = [s for s in dist_df.index if s.endswith(suffix) and s not in stations]
    for station in stations:
        valid_candidates = [c for c in candidates if c != station]
        distances = dist_df.loc[station, valid_candidates]
        nearest = distances.sort_values().head(k)
        results[station] = nearest    
    return results


def safe_name(text):
    return str(text).replace(" ", "_").replace("/", "_").replace("-", "_")

def make_outer_folds(df_case: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    if "date" in df_case.columns:
        df_case = df_case.sort_values("date").reset_index(drop=True)
    X_dummy = np.arange(len(df_case))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X_dummy))

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(rmse(y_true, y_pred), 2),
        "R2": round(r2_score(y_true, y_pred), 2),
    }


# =========================================================
# SCENARIOS
# =========================================================

def build_scenario_feature_sets(df: pd.DataFrame, target_col: str) -> Dict[str, List[str]]:
    temporal_vars = [c for c in TEMPORAL_VARS if c in df.columns]
    met_vars = [c for c in MET_VARS if c in df.columns]

    baseline_tm = temporal_vars + met_vars
    traffic_tm = baseline_tm + [c for c in TRAFFIC_VARS if c in df.columns]

    target_bkg0 = f"{target_col}_bkg0"

    nearest_bg_plus_traffic = traffic_tm + ([target_bkg0] if target_bkg0 in df.columns else [])

    aux_cols = [c for c in df.columns if ("_bkg" in c or "_trf" in c)]
    aux_extra_cols = [c for c in aux_cols if c != target_bkg0]

    multi_station_plus_traffic = nearest_bg_plus_traffic + aux_extra_cols

    nearest_bg_no_traffic = [c for c in nearest_bg_plus_traffic if c not in TRAFFIC_VARS]
    multi_station_no_traffic = [c for c in multi_station_plus_traffic if c not in TRAFFIC_VARS]

    feature_sets = {
        "baseline_tm": sorted(list(dict.fromkeys(baseline_tm))),
        "traffic_tm": sorted(list(dict.fromkeys(traffic_tm))),
        "nearest_bg_plus_traffic": sorted(list(dict.fromkeys(nearest_bg_plus_traffic))),
        "nearest_bg_no_traffic": sorted(list(dict.fromkeys(nearest_bg_no_traffic))),
        "multi_station_plus_traffic": sorted(list(dict.fromkeys(multi_station_plus_traffic))),
        "multi_station_no_traffic": sorted(list(dict.fromkeys(multi_station_no_traffic))),
    }

    return feature_sets

stations = ['LondonMaryleboneRoad', 'Camden-EustonRoad', 'Westminster-OxfordStreet', "CamdenKerbside", "Wandsworth-PutneyHighStreet"]
target_col = "NO2"

dist_df = pd.read_csv("distance_matrix_km_btw_stations.csv")
dist_df.set_index("Sitename_type", inplace=True)
dir_drive = r"Bkg_or_with_traffic"

info_stations = pd.read_csv(r"all_stations_more.csv")
dir_files = r"Datasets_to_train"
stations_considered = ['Camden Kerbside Urban Traffic','London Marylebone Road Urban Traffic','Westminster - Oxford Street Urban Traffic', 'Camden - Euston Road Urban Traffic','Wandsworth - Putney High Street Urban Traffic']
stationname_type = {'CamdenKerbside':'Camden Kerbside Urban Traffic','LondonMaryleboneRoad':'London Marylebone Road Urban Traffic','Westminster-OxfordStreet':'Westminster - Oxford Street Urban Traffic', 'Camden-EustonRoad':'Camden - Euston Road Urban Traffic','Wandsworth-PutneyHighStreet':'Wandsworth - Putney High Street Urban Traffic'}


all_fold_metrics_OK = []
all_oof_predictions_OK = []
all_fold_metrics_IDW = []
all_oof_predictions_IDW = []

variogram_params_list = []
target_col = "NO2"
final_outer_fold ="5"
for station_name in stations:

    filepath = Path(rf"Datasets_to_train_by_fold_pollutants\{target_col}\{station_name}\outerfold_{final_outer_fold}")

    df_case = pd.read_csv(os.path.join(dir_files, f"{station_name}.csv"))
    df_case["date"] = pd.to_datetime(df_case["date"])
    df_case = (df_case.sort_values("date").reset_index(drop=True).dropna(subset=[target_col]).reset_index(drop=True) )

    
    case_output_dir = OUTPUT_DIR / f"{safe_name(station_name)}_{safe_name(target_col)}"
    case_output_dir.mkdir(parents=True, exist_ok=True)

    feature_sets = build_scenario_feature_sets(df_case, target_col=target_col)

    nearest_background = find_nearest_by_type(dist_df,stations_considered, suffix="Background", k=2)
    nearest_traffic = find_nearest_by_type(dist_df, stations_considered, suffix="Traffic", k=2 )

    names_bkg = nearest_background[stationname_type[station_name]].index.tolist()
    clean_names_bkg = [name.split("Urban")[0].strip() for name in names_bkg]
    
    names_traffic = nearest_traffic[stationname_type[station_name]].index.tolist()
    clean_names_trf = [name.split("Urban")[0].strip() for name in names_traffic]
    print(clean_names_bkg, clean_names_trf)
    lons = [float(info_stations[info_stations["Site Name"]==clean_names_bkg[0]]["Longitude"].values[0])]
    lons.append(float(info_stations[info_stations["Site Name"]==clean_names_bkg[1]]["Longitude"].values[0]))
    lons.append(float(info_stations[info_stations["Site Name"]==clean_names_trf[0]]["Longitude"].values[0]))
    lons.append(float(info_stations[info_stations["Site Name"]==clean_names_trf[1]]["Longitude"].values[0]))		
    lats = [float(info_stations[info_stations["Site Name"]==clean_names_bkg[0]]["Latitude"].values[0])]
    lats.append(float(info_stations[info_stations["Site Name"]==clean_names_bkg[1]]["Latitude"].values[0]))
    lats.append(float(info_stations[info_stations["Site Name"]==clean_names_trf[0]]["Latitude"].values[0]))
    lats.append(float(info_stations[info_stations["Site Name"]==clean_names_trf[1]]["Latitude"].values[0]))		
    lon_target = float(info_stations[info_stations["Station"]==station_name]["Longitude"].values[0])
    lat_target = float(info_stations[info_stations["Station"]==station_name]["Latitude"].values[0])
    transformer = Transformer.from_crs(
        "EPSG:4326",   # longitude/latitude, WGS84
        "EPSG:27700",  # British National Grid
        always_xy=True
    )
    x_coords, y_coords = transformer.transform(lons, lats)
    x_target, y_target = transformer.transform(lon_target,  lat_target)
    fold_metrics = []
    oof_interpolations = []
    fold_metrics_IDW = []
    oof_interpolations_IDW = []
    
    x_train_path = filepath / (f"{target_col}_{station_name}_X_train_outerfold_{final_outer_fold}.csv")
    y_train_path = filepath / (f"{target_col}_{station_name}_y_train_outerfold_{final_outer_fold}.csv")
    x_test_path = filepath / ( f"{target_col}_{station_name}_X_test_outerfold_{final_outer_fold}.csv")
    y_test_path = filepath / ( f"{target_col}_{station_name}_y_test_outerfold_{final_outer_fold}.csv")

    X_test = pd.read_csv(x_test_path)
    final_test_dates = X_test["date"].copy()
    final_test_row_ids = ( X_test["row_id_original"].copy())
    y_test_outer = pd.read_csv(y_test_path)[target_col]
    feature_cols = feature_sets["multi_station_no_traffic"]
    X_test_outer = X_test[feature_cols].copy()

    # =========================================================
    # IDW interpolation
    # =========================================================

    coords_X = np.column_stack([
        x_coords,
        y_coords
    ])

    target = np.array([
        x_target,
        y_target
    ])

    # Euclidean distances in metres (EPSG:27700)
    dist = np.sqrt(
        ((coords_X - target) ** 2).sum(axis=1)
    )

    # Safety check: target must not coincide with
    # one of the auxiliary stations
    if np.any(dist == 0):
        raise ValueError(
            f"{station_name}: target coincides with "
            f"an auxiliary station."
        )

    # IDW power parameter
    p = 2

    weights = 1.0 / (dist ** p)
    weights = weights / weights.sum()

    aux_cols = [
        f"{target_col}_bkg0",
        f"{target_col}_bkg1",
        f"{target_col}_trf0",
        f"{target_col}_trf1",
    ]

    values_IDW = (
        X_test_outer[aux_cols]
        .apply(pd.to_numeric, errors="coerce")
    )

    # Require all four auxiliary stations
    valid_aux_mask = (
        values_IDW
        .notna()
        .all(axis=1)
    )

    pred_interp = np.full(
        len(values_IDW),
        np.nan,
        dtype=float
    )

    pred_interp[valid_aux_mask] = (
        values_IDW
        .loc[valid_aux_mask]
        .to_numpy(dtype=float)
        @ weights
    )

    n_aux_points = (
        values_IDW
        .notna()
        .sum(axis=1)
        .to_numpy()
    )

    # =========================================================
    # Evaluation
    # =========================================================

    y_true = np.asarray(
        y_test_outer,
        dtype=float
    )

    valid_mask = (
        np.isfinite(y_true)
        & np.isfinite(pred_interp)
    )

    metrics_IDW = compute_metrics(
        y_true[valid_mask],
        pred_interp[valid_mask]
    )

    print(
        f"{station_name} | "
        f"IDW p={p} | "
        f"evaluated={valid_mask.sum()}/{len(valid_mask)} | "
        f"RMSE={metrics_IDW['RMSE']:.2f} | "
        f"MAE={metrics_IDW['MAE']:.2f} | "
        f"R2={metrics_IDW['R2']:.2f}"
    )

    # =========================================================
    # Predictions
    # =========================================================

    df_predictions_IDW = pd.DataFrame({
        "station": station_name,
        "pollutant": target_col,
        "scenario": "Interpolation",
        "model_name": "IDW",
        "outer_fold": final_outer_fold,
        "split_role": "final_test_interpolation",
        "row_id_original": final_test_row_ids.to_numpy(),
        "date": final_test_dates.to_numpy(),
        "y_true": y_true,
        "y_pred": pred_interp,
        "n_aux_points": n_aux_points,
        "idw_power": p,
    })

    # =========================================================
    # Metrics
    # =========================================================

    df_metrics_IDW = pd.DataFrame([{
        "station": station_name,
        "pollutant": target_col,
        "scenario": "Interpolation",
        "model_name": "IDW",
        "outer_fold": final_outer_fold,
        "idw_power": p,
        "n_test": len(X_test),
        "n_test_evaluated": int(valid_mask.sum()),
        "n_test_excluded": int((~valid_mask).sum()),
        "MAE": metrics_IDW["MAE"],
        "RMSE": metrics_IDW["RMSE"],
        "R2": metrics_IDW["R2"],
    }])

    # =========================================================
    # Save
    # =========================================================

    df_predictions_IDW.to_csv(
        case_output_dir
        / f"final_test_predictions_IDW_{station_name}_{target_col}.csv",
        index=False
    )

    df_metrics_IDW.to_csv(
        case_output_dir
        / f"final_test_metrics_IDW_{station_name}_{target_col}.csv",
        index=False
    )