# This file is part of the article:
# "Leveraging Remote Traffic Data for Local Air Pollutant Estimation:
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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import lightgbm as lgb
from xgboost import XGBRegressor
import json
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from utils import *

# =========================================================
# SETTINGS
# =========================================================

N_OUTER_SPLITS = 5
RANDOM_STATE = 754

source_station = "LondonMaryleboneRoad"
pollutant = "NO2"

stations_validation = [
    "Camden-EustonRoad",
    "Westminster-OxfordStreet",
]

dir_files = Path(r"Datasets_to_train")

nested_dir = dir_files / "nested_cv_results_pollutants"
OUTPUT_DIR = Path("validation_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# =========================================================
# BASIC FUNCTIONS
# =========================================================

def safe_name(text):
    return str(text).replace(" ", "_").replace("/", "_").replace("-", "_")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def make_outer_folds(df_case, n_splits=5):
    df_case = df_case.sort_values("date").reset_index(drop=True)
    X_dummy = np.arange(len(df_case))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X_dummy))


def parse_value(v):
    if pd.isna(v):
        return None

    if isinstance(v, str):
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        if v.lower() == "none":
            return None

        try:
            f = float(v)
            if f.is_integer():
                return int(f)
            return f
        except ValueError:
            return v

    if isinstance(v, np.generic):
        return v.item()

    return v


def cast_params(model_name, params):
    params = {k: parse_value(v) for k, v in params.items() if parse_value(v) is not None}

    if model_name in ["rf", "etr"]:
        int_params = [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_leaf_nodes",
        ]

        for p in int_params:
            if p in params and params[p] is not None:
                params[p] = int(params[p])

    if model_name == "lgbm":
        int_params = [
            "n_estimators",
            "num_leaves",
            "max_depth",
            "min_child_samples",
            "subsample_freq",
        ]

        for p in int_params:
            if p in params and params[p] is not None:
                params[p] = int(params[p])

    if model_name == "xgb":
        int_params = ["n_estimators", "max_depth"]

        for p in int_params:
            if p in params and params[p] is not None:
                params[p] = int(params[p])

    return params


def build_model(model_name, params, random_state=42):
    params = cast_params(model_name, params)

    if model_name == "rf":
        return RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1,
            **params
        )

    if model_name == "etr":
        return ExtraTreesRegressor(
            random_state=random_state,
            n_jobs=-1,
            **params
        )

    if model_name == "lgbm":
        return LGBMRegressor(
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
            objective="regression",
            **params
        )

    if model_name == "xgb":
        return XGBRegressor(
            random_state=random_state,
            n_jobs=-1,
            objective="reg:squarederror",
            tree_method="hist",
            **params
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


metadata_cols = [
    "station",
    "pollutant",
    "scenario",
    "model_name",
    "outer_fold",
    "inner_best_score",
]

base_dir = Path(r"Datasets_to_train\nested_cv_results_pollutants")
for target_station in stations_validation:

    print(f"\nTarget station: {target_station}")

    target_case_output_dir = OUTPUT_DIR / f"{safe_name(target_station)}_{safe_name(pollutant)}"
    target_case_output_dir.mkdir(parents=True, exist_ok=True)

    df_target = pd.read_csv(os.path.join(dir_files, f"{target_station}.csv"))
    df_target["date"] = pd.to_datetime(df_target["date"])
    df_target = (df_target.sort_values("date").reset_index(drop=True).dropna(subset=[pollutant]).reset_index(drop=True) )
    df_target["row_id_original"] = np.arange(len(df_target))
        

    scenarios_map = {
        "traffic_tm":"scen1", 
        "baseline_tm":"scen1_w/o_traf",
        "nearest_bg_plus_traffic":"scen2", 
        "nearest_bg_no_traffic":"scen2_w/o_traf",
        "multi_station_plus_traffic":"scen3", 
        "multi_station_no_traffic":"scen3_w/o_traf"
    }



    all_fold_metrics = []
    all_oof_predictions = []
    outer_fold = 5
    for scenario, scen_map in scenarios_map.items():
    
        test_metrics = pd.read_csv(os.path.join(base_dir,f"{safe_name(source_station)}_{safe_name(pollutant)}",f"final_test_metrics_{source_station}_{pollutant}.csv"))
        model_name = test_metrics[(test_metrics["scenario"]==scenario)].iloc[0]["selected_model_name"]       
        print(f"Scenario={scenario} | source model={model_name}")    
        df_case = pd.read_csv(os.path.join(dir_files, f"{source_station}.csv"))      
        feature_sets = build_scenario_feature_sets(df_case, target_col=pollutant)
        filepath = Path(f"Datasets_to_train_by_fold_pollutants/{pollutant}/{source_station}/outerfold_{outer_fold}")
        X_test_source = pd.read_csv(os.path.join(filepath, f"{pollutant}_{source_station}_X_test_outerfold_{outer_fold}.csv"))
        test_row_ids_source = X_test_source["row_id_original"].copy()
        feature_cols = feature_sets[scenario]
        case_dir = os.path.join(dir_files,"nested_cv_results_pollutants", f"{source_station}_{pollutant}")         
        model_dir = os.path.join(case_dir, "models_final")
        model, model_path = load_model(
            model_name=model_name,
            model_dir=model_dir,
            station_results=source_station,
            pollutant=pollutant,
            scenario=scenario
        )

        filepath = Path(f"Datasets_to_train_by_fold_pollutants/{pollutant}/{target_station}/outerfold_{outer_fold}")
        X_test_target = pd.read_csv(os.path.join(filepath, f"{pollutant}_{target_station}_X_test_outerfold_{outer_fold}.csv"))
        y_test_target = pd.read_csv(os.path.join(filepath, f"{pollutant}_{target_station}_y_test_outerfold_{outer_fold}.csv"))

        test_row_ids_target = X_test_target["row_id_original"]
        test_dates_target = X_test_target["date"]
        X_test_target = X_test_target[feature_cols]
        y_test_target = y_test_target[pollutant]
        
        
        y_pred = model.predict(X_test_target)
        metrics = compute_metrics(y_test_target, y_pred)
        all_fold_metrics.append({
            "station": target_station,
            "pollutant": pollutant,
            "scenario": scenario,
            "model_name": model_name,
            "prediction_type": "transfer_fold_model",
            "source_station": source_station,
            "outer_fold": outer_fold,
            "n_test_target": len(X_test_target),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
        })

        all_oof_predictions.append(pd.DataFrame({
            "station": target_station,
            "pollutant": pollutant,
            "scenario": scenario,
            "model_name": model_name,
            "prediction_type": "transfer_fold_model",
            "source_station": source_station,
            "outer_fold": outer_fold,
            "row_id_original": test_row_ids_target,
            "date": test_dates_target,
            "y_true": y_test_target,
            "y_pred": y_pred,
        }))

    df_fold_metrics_transfer = pd.DataFrame(all_fold_metrics)
    df_oof_predictions_transfer = pd.concat(all_oof_predictions, ignore_index=True)

    df_fold_metrics_transfer.to_csv(
        target_case_output_dir / f"final_metrics_transfer_{target_station}_{pollutant}.csv",
        index=False
    )

    df_oof_predictions_transfer.to_csv(
        target_case_output_dir / f"final_predictions_transfer_{target_station}_{pollutant}.csv",
        index=False
    )