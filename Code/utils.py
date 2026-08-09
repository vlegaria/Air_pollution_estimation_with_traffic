from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pykrige.ok import OrdinaryKriging
import re
from matplotlib import units
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

from xgboost import XGBRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
import joblib
dir_files = r"Datasets_to_train"
interpolation_dir = Path(r"interpolation_results")


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(rmse(y_true, y_pred), 2),
        "R2": round(r2_score(y_true, y_pred), 2),
    }


def safe_name(text):
    return str(text).replace(" ", "_").replace("/", "_").replace("-", "_")

def get_predictions_test_fold(stations, base_dir):
    
    scenarios_map = {
        "traffic_tm":"scen1", 
        "baseline_tm":"scen1_w/o_traf",
        "nearest_bg_plus_traffic":"scen2", 
        "nearest_bg_no_traffic":"scen2_w/o_traf",
        "multi_station_plus_traffic":"scen3", 
        "multi_station_no_traffic":"scen3_w/o_traf"
    }

    pollutants = ["NO2"]
    scenarios_to_compare = ["scen1", "scen3"]
    compare = []
    print("Comparing ML predictions with interpolations:")
    for scenario_to_compare in scenarios_to_compare:
        print(f"\n--- Scenario: {scenario_to_compare} ---")
        for pollutant in pollutants:
            for station in stations:
                print(pollutant, scenario_to_compare, station)

                test_metrics = pd.read_csv(os.path.join(base_dir,f"{safe_name(station)}_{safe_name(pollutant)}",f"final_test_metrics_{station}_{pollutant}.csv"))
                test_metrics["scenario"] = test_metrics["scenario"].apply(lambda x: scenarios_map.get(x, x))
                file_path = os.path.join(base_dir,f"{safe_name(station)}_{safe_name(pollutant)}")
                df_predictions = pd.read_csv(os.path.join(file_path,rf"final_test_predictions_{station}_{pollutant}.csv"))
                df_predictions["scenario"] = df_predictions["scenario"].apply(lambda x: scenarios_map.get(x, x))
                df_ml = df_predictions[ (df_predictions["scenario"] == scenario_to_compare)].copy()
                
                # ======================
                # ML predictions
                # ======================
                
                df_ml["scenario_eval"] = scenario_to_compare

                # ======================
                # OK interpolation
                # ======================
                df_ok = pd.read_csv(rf"interpolation_results_chronological_selection\{safe_name(station)}_{pollutant}\final_test_predictions_OK_{station}_{pollutant}.csv")
                
                df_compare_OK = df_ml.merge(
                    df_ok,
                    on=["station", "pollutant", "outer_fold", "row_id_original", "date", "y_true"],
                    suffixes=("_ml", "_ok")
                ).dropna(subset=["y_pred_ml", "y_pred_ok"])

                # ======================
                # IDW interpolation
                # ======================
                df_idw = pd.read_csv(rf"interpolation_results_IDW\{safe_name(station)}_{pollutant}\final_test_predictions_IDW_{station}_{pollutant}.csv")
                print(rf"interpolation_results_IDW\{safe_name(station)}_{pollutant}\final_test_predictions_IDW_{station}_{pollutant}.csv")      

                df_compare_IDW = df_ml.merge(
                    df_idw,
                    on=["station", "pollutant", "outer_fold", "row_id_original", "date", "y_true"],
                    suffixes=("_ml", "_idw")
                ).dropna(subset=["y_pred_ml", "y_pred_idw"])

                # ======================
                # Merge final (ML + IDW + OK)
                # ======================
                keys = ["station", "pollutant", "outer_fold", "row_id_original", "date", "y_true"]
                df_final = df_compare_IDW.merge(
                    df_compare_OK[keys + ["y_pred_ok"]],
                    on=keys,
                    how="inner"
                )

                df_final["scenario_eval"] = scenario_to_compare

                compare.append(df_final)

    df_compare_all = pd.concat(compare, ignore_index=True)
    return df_compare_all


def get_predictions(dir_files, interpolation_dir ):
    folds_files = list(Path(os.path.join(
                        dir_files,
                        "nested_cv_results_pollutants")).rglob("fold_metrics*.csv"))

    df_fold_metrics_all = pd.concat(
        [pd.read_csv(f) for f in folds_files],
        ignore_index=True
    )
    
    df_fold_metrics_all = df_fold_metrics_all[df_fold_metrics_all["model_name"] != "ridge"].copy()

    oof_predictions_files = list(Path(os.path.join(
                        dir_files,
                        "nested_cv_results_pollutants")).rglob("oof_predictions*.csv"))
    df_predictions = pd.concat(
        [pd.read_csv(f) for f in oof_predictions_files],
        ignore_index=True
    )

    oof_interpolations_files = list(Path(interpolation_dir).rglob("oof_predictions_OK_*.csv"))
    df_oof_interpolation_OK = pd.concat(
        [pd.read_csv(f) for f in oof_interpolations_files],
        ignore_index=True
    )

    oof_interpolations_files = list(Path(interpolation_dir).rglob("oof_predictions_IDW_*.csv"))
    df_oof_interpolation_IDW = pd.concat(
        [pd.read_csv(f) for f in oof_interpolations_files],
        ignore_index=True
    )

    scenarios_map = {
        "traffic_tm":"scen1", 
        "baseline_tm":"scen1_w/o_traf",
        "nearest_bg_plus_traffic":"scen2", 
        "nearest_bg_no_traffic":"scen2_w/o_traf",
        "multi_station_plus_traffic":"scen3", 
        "multi_station_no_traffic":"scen3_w/o_traf"
    }

    df_predictions["scenario"] = df_predictions["scenario"].apply(lambda x: scenarios_map.get(x, x))

    df_perf = (
        df_fold_metrics_all
        .groupby(["station", "pollutant", "scenario", "model_name"])
        .agg(
            RMSE_mean=("RMSE", "mean")).reset_index())
    df_best_model = df_perf.groupby(
        ["station", "pollutant", "scenario", "model_name"]
    )["RMSE_mean"].mean().reset_index()

    df_best_model = df_best_model.loc[
        df_best_model.groupby(
            ["station", "pollutant", "scenario"]
        )["RMSE_mean"].idxmin()]
    df_best_model["scenario"] = df_best_model["scenario"].apply(lambda x: scenarios_map.get(x, x))


    pollutants = ["NO2"]
    stations = ["LondonMaryleboneRoad", "CamdenKerbside", "Wandsworth-PutneyHighStreet"]
    scenarios_to_compare = ["scen1", "scen3"]
    compare = []
    print("Comparing ML predictions with interpolations:")
    for scenario_to_compare in scenarios_to_compare:
        print(f"\n--- Scenario: {scenario_to_compare} ---")

        for pollutant in pollutants:
            for station in stations:

                print(pollutant, scenario_to_compare, station)

                match = df_best_model[
                    (df_best_model["station"] == station) &
                    (df_best_model["pollutant"] == pollutant) & 
                    (df_best_model["scenario"] == scenario_to_compare)
                ]

                if match.empty:
                    print("No best model found, skipping...")
                    continue

                model_name = match.iloc[0]["model_name"]

                # ======================
                # ML predictions
                # ======================
                df_ml = df_predictions[
                    (df_predictions["station"] == station) &
                    (df_predictions["pollutant"] == pollutant) &
                    (df_predictions["scenario"] == scenario_to_compare) &
                    (df_predictions["model_name"] == model_name)
                ].copy()

                df_ml["scenario_eval"] = scenario_to_compare

                # ======================
                # OK interpolation
                # ======================
                df_ok = df_oof_interpolation_OK[
                    (df_oof_interpolation_OK["station"] == station) &
                    (df_oof_interpolation_OK["pollutant"] == pollutant)
                ].copy()

                df_compare_OK = df_ml.merge(
                    df_ok,
                    on=["station", "pollutant", "outer_fold", "row_id_original", "date", "y_true"],
                    suffixes=("_ml", "_ok")
                ).dropna(subset=["y_pred_ml", "y_pred_ok"])

                # ======================
                # IDW interpolation
                # ======================
                df_idw = df_oof_interpolation_IDW[
                    (df_oof_interpolation_IDW["station"] == station) &
                    (df_oof_interpolation_IDW["pollutant"] == pollutant)
                ].copy()

                df_compare_IDW = df_ml.merge(
                    df_idw,
                    on=["station", "pollutant", "outer_fold", "row_id_original", "date", "y_true"],
                    suffixes=("_ml", "_idw")
                ).dropna(subset=["y_pred_ml", "y_pred_idw"])

                # ======================
                # Merge final (ML + IDW + OK)
                # ======================
                keys = ["station", "pollutant", "outer_fold", "row_id_original", "date", "y_true"]
                df_final = df_compare_IDW.merge(
                    df_compare_OK[keys + ["y_pred_ok"]],
                    on=keys,
                    how="inner"
                )

                df_final["scenario_eval"] = scenario_to_compare

                compare.append(df_final)

    df_compare_all = pd.concat(compare, ignore_index=True)
    return df_compare_all
def fold_metric_summary(df, pred_col, scenario=None, station=None, pollutant=None):

    df = df.copy()

    if scenario is not None:
        df = df[df["scenario_eval"] == scenario]

    if station is not None:
        df = df[df["station"] == station]

    if pollutant is not None:
        df = df[df["pollutant"] == pollutant]
    rows = []

    for fold, g in df.groupby("outer_fold"):
        g = g.dropna(subset=["y_true", pred_col])
        if scenario is not None:
            g = g[g["scenario_ml"] == scenario]

        rows.append({
            "outer_fold": fold,
            "RMSE": rmse(g["y_true"], g[pred_col]),
            "MAE": mean_absolute_error(g["y_true"], g[pred_col]),
            "R2": r2_score(g["y_true"], g[pred_col]),
            "n": len(g),
        })

    df_folds = pd.DataFrame(rows)

    return {
        "RMSE_mean": df_folds["RMSE"].mean(),
        "RMSE_std": df_folds["RMSE"].std(),
        "MAE_mean": df_folds["MAE"].mean(),
        "MAE_std": df_folds["MAE"].std(),
        "R2_mean": df_folds["R2"].mean(),
        "R2_std": df_folds["R2"].std(),
        "n_total": df_folds["n"].sum(),
    }



N_OUTER_SPLITS = 5
RANDOM_STATE = 754
TRAFFIC_VARS = [ "traffic_level", "currenttraveltime"]
TEMPORAL_VARS = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos"]

MET_VARS = [
    "temp_ow",
    "pressure_ow",
    "humidity_ow",
    "clouds_ow",
    "wind_speed_ow",
    "wdr_sin", "wdr_cos",
    "dew_point_ow",
]


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



def load_model(model_name, model_dir, station_results, pollutant, scenario):
    if model_name == "xgb":
        model_path = os.path.join(
            model_dir,
            f"{model_name}_{station_results}_{pollutant}_{scenario}_final.json"
        )
        model = XGBRegressor()
        model.load_model(model_path)

    elif model_name == "lgbm":
        model_path = os.path.join(
            model_dir,
            f"{model_name}_{station_results}_{pollutant}_{scenario}_final.txt"
        )
        model = lgb.Booster(model_file=model_path)

    elif model_name in ["etr", "rf"]:
        model_path = os.path.join(
            model_dir,
            f"{model_name}_{station_results}_{pollutant}_{scenario}_final.joblib"
        )
        model = joblib.load(model_path)

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model, model_path