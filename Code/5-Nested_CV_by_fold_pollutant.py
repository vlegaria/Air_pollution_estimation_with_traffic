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
import os
import warnings
import json
import time
import zipfile
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import argparse

#warnings.filterwarnings("ignore")
warnings.filterwarnings("default")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 100
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3
RANDOM_STATE = 754
TUNING_METRIC = "RMSE"


OUTPUT_DIR = Path("nested_cv_results_pollutants")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


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


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

def safe_name(text):
    return str(text).replace(" ", "_").replace("/", "_").replace("-", "_")

def save_model(
    model,
    model_name: str,
    station_name: str,
    pollutant_name: str,
    scenario_name: str,
    output_dir,
    outer_fold=None
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    station = safe_name(station_name)
    pollutant = safe_name(pollutant_name)
    scenario = safe_name(scenario_name)

    fold_part = f"_fold{outer_fold}" if outer_fold is not None else "_final"
    base = f"{model_name}_{station}_{pollutant}_{scenario}{fold_part}"

    if model_name == "lgbm":
        path = output_dir / f"{base}.txt"
        model.booster_.save_model(str(path))

    elif model_name == "xgb":
        path = output_dir / f"{base}.json"
        model.save_model(str(path))

    else:
        path = output_dir / f"{base}.joblib"
        joblib.dump(model, path)

    return path




# =========================================================
# SEARCH SPACES
# =========================================================
def cast_params(model_name, params):
    params = params.copy()

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
            if p in params:
                params[p] = int(params[p])

    if model_name == "xgb":
        int_params = [
            "n_estimators",
            "max_depth",
        ]
        for p in int_params:
            if p in params:
                params[p] = int(params[p])

    return params

def suggest_params_ridge(trial):
    return {"alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True )}
    
def suggest_params_rf(trial):
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9]
        ),
        "bootstrap": bootstrap,
        "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-6, 1e-2, log=True),
    }

    if bootstrap:
        params["max_samples"] = trial.suggest_float("max_samples", 0.5, 0.95)

    return params


def suggest_params_etr(trial):
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 35),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 20, 300),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9]
        ),
        "bootstrap": bootstrap,
        "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-6, 1e-2, log=True),
    }

    if bootstrap:
        params["max_samples"] = trial.suggest_float("max_samples", 0.5, 0.95)

    return params


def suggest_params_lgbm(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 1.0, log=True),
    }


def suggest_params_xgb(trial):
    booster = "gbtree"

    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "booster": booster,
    }


def suggest_params(trial, model_name: str):
    if model_name == "ridge":
        return suggest_params_ridge(trial)
    if model_name == "rf":
        return suggest_params_rf(trial)
    if model_name == "etr":
        return suggest_params_etr(trial)
    if model_name == "lgbm":
        return suggest_params_lgbm(trial)
    if model_name == "xgb":
        return suggest_params_xgb(trial)

    raise ValueError(f"Modelo no soportado: {model_name}")


def build_model(model_name: str, params: dict, random_state: int = 42):
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

    
    if model_name == "ridge":
        return Pipeline([ (
                "imputer", SimpleImputer(strategy="median",add_indicator=False  ) ),
            ("scaler", StandardScaler() ), ( "model",  Ridge( alpha=float(params["alpha"]) ) ) ])

    raise ValueError(f"Model incorrect: {model_name}")


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

# =========================================================
# FOLDS
# =========================================================

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



# =========================================================
# OPTUNA OBJECTIVE
# =========================================================

def optuna_objective(
    trial,
    feature_cols: List,
    outer_fold: int,
    station_name: str,
    target_col: str,
    model_name: str,
    metric: str,
    random_state: int = 42,
):
    params = suggest_params(trial, model_name)

    scores = []
    for inner_fold in range(1, N_INNER_SPLITS + 1):
        filepath = Path(f"Datasets_to_train_by_fold_pollutants/{target_col}/{station_name}/outerfold_{outer_fold}/innerfold")
        X_tr = pd.read_csv(os.path.join(filepath, f"{target_col}_{station_name}_X_tr_outerfold_{outer_fold}_innerfold_{inner_fold}.csv"))
        y_tr = pd.read_csv(os.path.join(filepath, f"{target_col}_{station_name}_y_tr_outerfold_{outer_fold}_innerfold_{inner_fold}.csv"))
        X_val = pd.read_csv(os.path.join(filepath, f"{target_col}_{station_name}_X_val_outerfold_{outer_fold}_innerfold_{inner_fold}.csv"))
        y_val = pd.read_csv(os.path.join(filepath, f"{target_col}_{station_name}_y_val_outerfold_{outer_fold}_innerfold_{inner_fold}.csv"))

        X_tr = X_tr[feature_cols].copy()   
        y_tr = y_tr[target_col]
        X_val = X_val[feature_cols].copy()   
        y_val = y_val[target_col]
        
        model = build_model(model_name, params, random_state=random_state)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        if metric.upper() == "MAE":
            score = mean_absolute_error(y_val, y_pred)
        elif metric.upper() == "RMSE":
            score = rmse(y_val, y_pred)
        else:
            raise ValueError("metric must be 'MAE' or 'RMSE'")

        scores.append(score)

        intermediate_score = float(np.mean(scores))
        trial.report(intermediate_score, step=inner_fold)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


# =========================================================
# NESTED CV
# =========================================================

def run_nested_cv_for_scenario_model(
    target_col: str,
    feature_cols: List[str],
    scenario_name: str,
    model_name: str,
    station_name: str,
    pollutant_name: str,
):

    fold_metrics = []
    oof_predictions = []
    best_params_records = []
    print(feature_cols)
    for outer_fold in range(1, N_OUTER_SPLITS):
        
    
        print(f"Running nested CV fold {outer_fold}/{N_OUTER_SPLITS - 1}", flush=True)
        filepath = Path(f"Datasets_to_train_by_fold_pollutants/{target_col}/{station_name}/outerfold_{outer_fold}")
        X_train_outer = pd.read_csv(os.path.join(filepath, f"{target_col}_{station_name}_X_train_outerfold_{outer_fold}.csv"))
        y_train_outer = pd.read_csv(os.path.join(filepath, f"{target_col}_{station_name}_y_train_outerfold_{outer_fold}.csv"))
        X_test_outer = pd.read_csv(os.path.join(filepath, f"{target_col}_{station_name}_X_test_outerfold_{outer_fold}.csv"))
        y_test_outer = pd.read_csv(os.path.join(filepath, f"{target_col}_{station_name}_y_test_outerfold_{outer_fold}.csv"))

        test_row_ids = X_test_outer["row_id_original"].copy()
        test_dates = X_test_outer["date"].copy()

        X_train_outer = X_train_outer[feature_cols].copy()
        y_train_outer = y_train_outer[target_col]
        X_test_outer = X_test_outer[feature_cols].copy()
        y_test_outer = y_test_outer[target_col]
       

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=1
            )
        )

        study.optimize(
            lambda trial: optuna_objective(
                trial=trial,
                feature_cols = feature_cols,
                outer_fold=outer_fold,
                station_name = station_name,
                target_col = target_col,
                model_name=model_name,
                metric=TUNING_METRIC,
                random_state=RANDOM_STATE
            ),
            n_trials=N_TRIALS,
            show_progress_bar=False
        )

        best_params = study.best_params
        best_score_inner = study.best_value

        final_model = build_model(model_name, best_params, random_state=RANDOM_STATE)        
        final_model.fit(X_train_outer, y_train_outer)

        y_pred = final_model.predict(X_test_outer)
        metrics = compute_metrics(y_test_outer, y_pred)

        fold_metrics.append({
            "station": station_name,
            "pollutant": pollutant_name,
            "scenario": scenario_name,
            "model_name": model_name,
            "outer_fold": outer_fold,
            "n_train": len(X_train_outer),
            "n_test": len(X_test_outer),
            "inner_best_score": best_score_inner,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
        })
        oof_predictions.append(pd.DataFrame({
            "station": station_name,
            "pollutant": pollutant_name,
            "scenario": scenario_name,
            "model_name": model_name,
            "outer_fold": outer_fold,
            "split_role": "development_oof",
            "row_id_original": test_row_ids.to_numpy(), 
            "date": test_dates.values, 
            "y_true": y_test_outer.values,
            "y_pred": y_pred,
        }))

        best_params_record = {
            "station": station_name,
            "pollutant": pollutant_name,
            "scenario": scenario_name,
            "model_name": model_name,
            "outer_fold": outer_fold,
            "inner_best_score": best_score_inner,
        }

        best_params_record.update(best_params)
        best_params_records.append(best_params_record)

        print(
            f"[{station_name} | {pollutant_name} | {scenario_name} | {model_name}] "
            f"fold {outer_fold}/{N_OUTER_SPLITS - 1} | "
            f"MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | R2={metrics['R2']:.4f}", flush=True
        )
    return (
        pd.DataFrame(fold_metrics),
        pd.concat(oof_predictions, ignore_index=True),
        pd.DataFrame(best_params_records)
    )


def validate_prediction_frame(
    prediction_df: pd.DataFrame,
    context: str,
):
    required = [
        "station",
        "pollutant",
        "scenario",
        "model_name",
        "outer_fold",
        "row_id_original",
        "date",
        "y_true",
        "y_pred",
    ]

    missing = [
        col for col in required
        if col not in prediction_df.columns
    ]

    if missing:
        raise ValueError(
            f"{context}: missing columns {missing}"
        )

    if prediction_df[required].isna().any().any():
        missing_counts = (
            prediction_df[required]
            .isna()
            .sum()
        )
        missing_counts = missing_counts[
            missing_counts > 0
        ]

        raise ValueError(
            f"{context}: NaNs in prediction output:\n"
            f"{missing_counts}"
        )

    duplicated = prediction_df.duplicated(
        subset=[
            "station",
            "pollutant",
            "scenario",
            "model_name",
            "outer_fold",
            "row_id_original",
        ]
    )

    if duplicated.any():
        raise ValueError(
            f"{context}: duplicated prediction rows detected."
        )
# =========================================================
# FINAL MODEL
# =========================================================

def select_best_algorithm_per_scenario(
    df_fold_metrics: pd.DataFrame,
    selection_metric: str = "RMSE",
    excluded_models=("ridge",),
) -> pd.DataFrame:
    """Select one algorithm per scenario using folds 1--4 only."""
    required = {
        "station", "pollutant", "scenario", "model_name",
        "outer_fold", selection_metric,
    }
    missing = required.difference(df_fold_metrics.columns)
    if missing:
        raise ValueError(f"Missing columns for model selection: {sorted(missing)}")

    development = df_fold_metrics[df_fold_metrics["outer_fold"] < N_OUTER_SPLITS].copy()

    if excluded_models:
        development = development[
            ~development["model_name"].isin(excluded_models)
        ].copy()

    summary = (
        development
        .groupby(
            ["station", "pollutant", "scenario", "model_name"],
            as_index=False,
        )
        .agg(
            selection_score_mean=(selection_metric, "mean"),
            selection_score_sd=(selection_metric, "std"),
            n_development_folds=("outer_fold", "nunique"),
        )
    )

    selected = (
        summary
        .sort_values(
            ["station", "pollutant", "scenario", "selection_score_mean", "model_name"]
        )
        .groupby(["station", "pollutant", "scenario"], as_index=False)
        .first()
    )
    selected["selection_metric"] = selection_metric
    return selected


def tune_fit_and_evaluate_final_model(
    target_col: str,
    feature_cols: List[str],
    station_name: str,
    pollutant_name: str,
    scenario_name: str,
    model_name: str,
    output_dir,
    final_outer_fold: int = N_OUTER_SPLITS,
):
    """Retune on fold-5 train, fit there, and evaluate once on fold-5 test.

    The fold-5 test data are never used for algorithm or hyperparameter
    selection. They are accessed only after both choices are complete.
    """
    feature_cols = list(feature_cols)

    filepath = Path(f"Datasets_to_train_by_fold_pollutants/{target_col}/{station_name}/outerfold_{final_outer_fold}")
                
    print("Fiting best model")
    x_train_path = filepath / (f"{target_col}_{station_name}_X_train_outerfold_{final_outer_fold}.csv")
    y_train_path = filepath / (f"{target_col}_{station_name}_y_train_outerfold_{final_outer_fold}.csv")
    x_test_path = filepath / ( f"{target_col}_{station_name}_X_test_outerfold_{final_outer_fold}.csv")
    y_test_path = filepath / ( f"{target_col}_{station_name}_y_test_outerfold_{final_outer_fold}.csv")

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    final_test_dates = X_test["date"].copy()
    final_test_row_ids = ( X_test["row_id_original"].copy())
    y_train = pd.read_csv(y_train_path)[target_col]
    y_test = pd.read_csv(y_test_path)[target_col]

    X_train = X_train[feature_cols].copy()
    X_test = X_test[feature_cols].copy()

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=1,
        ),
    )
    study.optimize(
        lambda trial: optuna_objective(
            trial=trial,
            feature_cols = feature_cols,
            outer_fold= final_outer_fold,
            target_col= target_col,
            station_name = station_name,
            model_name=model_name,
            metric=TUNING_METRIC,
            random_state=RANDOM_STATE,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    final_best_params = study.best_params
    final_inner_score = float(study.best_value)

    final_model = build_model(
        model_name,
        final_best_params,
        random_state=RANDOM_STATE,
    )
    final_model.fit(X_train, y_train)

    # First and only use of the untouched chronological final test set.
    y_pred_final = final_model.predict(X_test)
    final_metrics = compute_metrics(y_test, y_pred_final)

    model_path = save_model(
        model=final_model,
        model_name=model_name,
        station_name=station_name,
        pollutant_name=pollutant_name,
        scenario_name=scenario_name,
        output_dir=output_dir,
        outer_fold=None,
    )

    metric_record = {
        "station": station_name,
        "pollutant": pollutant_name,
        "scenario": scenario_name,
        "selected_model_name": model_name,
        "final_outer_fold": final_outer_fold,
        "n_development": len(X_train),
        "n_final_test": len(X_test),
        "final_tuning_metric": TUNING_METRIC,
        "final_inner_best_score": final_inner_score,
        "Final_test_MAE": final_metrics["MAE"],
        "Final_test_RMSE": final_metrics["RMSE"],
        "Final_test_R2": final_metrics["R2"],
        "model_path": str(model_path),
    }

    param_record = {
        "station": station_name,
        "pollutant": pollutant_name,
        "scenario": scenario_name,
        "selected_model_name": model_name,
        "final_outer_fold": final_outer_fold,
        "final_inner_best_score": final_inner_score,
        **final_best_params,
    }

    if final_test_dates is None:
        final_test_dates = pd.Series(pd.NaT, index=np.arange(len(y_test)))

    if model_name == "ridge":
        split_role = "final_test_ridge_baseline"
    else:
        split_role = "final_test_selected_model"
    prediction_df = pd.DataFrame({
        "station": station_name,
        "pollutant": pollutant_name,
        "scenario": scenario_name,
        "model_name": model_name,
        "outer_fold": final_outer_fold,
        "split_role": split_role,
        "row_id_original": final_test_row_ids.to_numpy(),
        "date": final_test_dates.to_numpy(),
        "y_true": y_test.to_numpy(),
        "y_pred": y_pred_final,
    })

    return (
        final_model,
        model_path,
        metric_record,
        param_record,
        prediction_df,
    )




# =========================================================
# SAVE RESULTS
# =========================================================
def save_experiment_metadata(
    output_dir,
    station_name,
    pollutant_name,
    target_col,
    model_names,
    feature_sets,
    outer_folds
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"feature_sets_{station_name}_{target_col}.json", "w") as f:
        json.dump(feature_sets, f, indent=2)

    outer_folds_serializable = [
        {
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
        }
        for train_idx, test_idx in outer_folds
    ]

    with open(output_dir / f"outer_folds_{station_name}_{target_col}.json", "w") as f:
        json.dump(outer_folds_serializable, f, indent=2)

    config = {
        "n_trials": N_TRIALS,
        "outer_splits": N_OUTER_SPLITS,
        "inner_splits": N_INNER_SPLITS,
        "tuning_metric": TUNING_METRIC,
        "random_state": RANDOM_STATE,
        "models": model_names,
        "station": station_name,
        "pollutant": pollutant_name,
        "target_col": target_col,
        "traffic_vars": TRAFFIC_VARS,
        "temporal_vars": TEMPORAL_VARS,
        "met_vars": MET_VARS,
    }

    with open(output_dir / f"experiment_config_{station_name}_{target_col}.json", "w") as f:
        json.dump(config, f, indent=2)

def zip_results(output_dir, station_name, pollutant_name, target_col):
    output_dir = Path(output_dir)

    files_to_zip = [
        output_dir / f"fold_metrics_{station_name}_{pollutant_name}.csv",
        output_dir / f"oof_predictions_{station_name}_{pollutant_name}.csv",
        output_dir / f"best_params_{station_name}_{pollutant_name}.csv",
        output_dir / f"stats_{station_name}_{pollutant_name}.csv",
        output_dir / f"final_models_{station_name}_{pollutant_name}.csv",
        output_dir / f"feature_sets_{station_name}_{target_col}.json",
        output_dir / f"outer_folds_{station_name}_{target_col}.json",
        output_dir / f"experiment_config_{station_name}_{target_col}.json",
    ]

    zip_path = output_dir / f"results_bundle_{station_name}_{target_col}.zip"

    with zipfile.ZipFile(zip_path, "w") as z:
        for file_path in files_to_zip:
            if file_path.exists():
                z.write(file_path, arcname=file_path.name)

    return zip_path

# =========================================================
# MAIN PIPELINE
# =========================================================
def run_case(
    df_case: pd.DataFrame,
    station_name: str,
    pollutant_name: str,
    target_col: str,
    model_names: List[str],
    scenario_names: List[str] = None,
):
    df_case = df_case.copy()
    df_case["date"] = pd.to_datetime(df_case["date"])
    df_case = df_case.sort_values("date").reset_index(drop=True)

    case_output_dir = OUTPUT_DIR / f"{safe_name(station_name)}_{safe_name(pollutant_name)}"
    case_output_dir.mkdir(parents=True, exist_ok=True)

    feature_sets = build_scenario_feature_sets(df_case, target_col=target_col)

    if scenario_names is None:
        scenario_names = [
            "baseline_tm",
            "traffic_tm",
            "nearest_bg_plus_traffic",
            "nearest_bg_no_traffic",    
            "multi_station_plus_traffic",
            "multi_station_no_traffic",
        ]

    outer_folds = make_outer_folds(df_case, n_splits=N_OUTER_SPLITS)

    all_fold_metrics = []
    all_oof_predictions = []
    all_best_params = []

    for scenario_name in scenario_names:
        feature_cols = feature_sets[scenario_name]

        for model_name in model_names:
            print(f"Scenario: {scenario_name} | Model: {model_name}", flush=True)
            print(f"Features: {feature_cols}", flush=True)

            df_metrics, df_oof, df_params = run_nested_cv_for_scenario_model(
                target_col=target_col,
                feature_cols=feature_cols,
                scenario_name=scenario_name,
                model_name=model_name,
                station_name=station_name,
                pollutant_name=pollutant_name,
            )

            all_fold_metrics.append(df_metrics)
            all_oof_predictions.append(df_oof)
            all_best_params.append(df_params)

    df_fold_metrics_all = pd.concat(all_fold_metrics, ignore_index=True)
    df_oof_predictions_all = pd.concat(all_oof_predictions, ignore_index=True)
    df_best_params_all = pd.concat(all_best_params, ignore_index=True)

    validate_prediction_frame(
        df_oof_predictions_all,
        context="Development OOF predictions"
    )

    df_fold_metrics_all.to_csv(
        case_output_dir / f"fold_metrics_{station_name}_{pollutant_name}.csv",
        index=False
    )

    df_oof_predictions_all.to_csv(
        case_output_dir / f"oof_predictions_{station_name}_{pollutant_name}.csv",
        index=False
    )

    df_best_params_all.to_csv(
        case_output_dir / f"best_params_{station_name}_{pollutant_name}.csv",
        index=False
    )


    # -----------------------------------------------------
    # Algorithm selection uses development folds 1-4 only.
    # -----------------------------------------------------
    df_selected_models = select_best_algorithm_per_scenario(
        df_fold_metrics_all,
        selection_metric=TUNING_METRIC,
        excluded_models = ("ridge",),
    )
    df_selected_models.to_csv(
        case_output_dir / f"selected_models_{station_name}_{pollutant_name}.csv",
        index=False,
    )

    final_metric_records = []
    final_param_records = []
    final_prediction_frames = []

    for _, selected_row in df_selected_models.iterrows():
        scenario_name = selected_row["scenario"]
        selected_model_name = selected_row["model_name"]
        feature_cols = feature_sets[scenario_name]

        print(
            f"Final chronological evaluation | scenario={scenario_name} | "
            f"selected model={selected_model_name}",
            flush=True,
        )

        (
            _,
            model_path,
            metric_record,
            param_record,
            prediction_df,
        ) = tune_fit_and_evaluate_final_model(
            target_col=target_col,
            feature_cols=feature_cols,
            station_name=station_name,
            pollutant_name=pollutant_name,
            scenario_name=scenario_name,
            model_name=selected_model_name,
            output_dir=case_output_dir / "models_final",
            final_outer_fold=N_OUTER_SPLITS,
        )

        metric_record.update({
            "development_selection_score_mean": selected_row["selection_score_mean"],
            "development_selection_score_sd": selected_row["selection_score_sd"],
            "development_selection_metric": selected_row["selection_metric"],
        })
        metric_record["model_path"] = str(model_path)

        final_metric_records.append(metric_record)
        final_param_records.append(param_record)
        final_prediction_frames.append(prediction_df)

    df_final_test_metrics = pd.DataFrame(final_metric_records)
    df_final_best_params = pd.DataFrame(final_param_records)
    df_final_test_predictions = (
        pd.concat(final_prediction_frames, ignore_index=True)
        if final_prediction_frames else pd.DataFrame()
    )

    validate_prediction_frame(
        df_final_test_predictions,
        context="Final selected-model predictions"
    )
    df_final_test_metrics.to_csv(
        case_output_dir / f"final_test_metrics_{station_name}_{pollutant_name}.csv",
        index=False,
    )
    df_final_best_params.to_csv(
        case_output_dir / f"final_best_params_{station_name}_{pollutant_name}.csv",
        index=False,
    )
    df_final_test_predictions.to_csv(
        case_output_dir / f"final_test_predictions_{station_name}_{pollutant_name}.csv",
        index=False,
    )


    ridge_metric_records = []
    ridge_param_records = []
    ridge_prediction_frames = []

    for scenario_name in scenario_names:

        feature_cols = feature_sets[scenario_name]

        print( f"Final Ridge baseline evaluation | "
            f"scenario={scenario_name}", flush=True )

        (
            ridge_model,
            ridge_model_path,
            ridge_metric_record,
            ridge_param_record,
            ridge_prediction_df,
        ) = tune_fit_and_evaluate_final_model(
            target_col=target_col,
            feature_cols=feature_cols,
            station_name=station_name,
            pollutant_name=pollutant_name,
            scenario_name=scenario_name,
            model_name="ridge",
            output_dir=case_output_dir / "models_final_ridge",
            final_outer_fold=N_OUTER_SPLITS,
        )

        ridge_metric_records.append(ridge_metric_record)

        ridge_param_records.append(ridge_param_record)

        ridge_prediction_frames.append(ridge_prediction_df)

    df_ridge_final_metrics = pd.DataFrame(ridge_metric_records)

    df_ridge_final_params = pd.DataFrame(ridge_param_records)

    df_ridge_final_predictions = pd.concat(ridge_prediction_frames,ignore_index=True )

    
    validate_prediction_frame(
        df_ridge_final_predictions,
        context="Final Ridge predictions"
    )

    df_ridge_final_metrics.to_csv( case_output_dir / f"ridge_final_test_metrics_{station_name}_{pollutant_name}.csv", index=False)

    df_ridge_final_params.to_csv(case_output_dir / f"ridge_final_best_params_{station_name}_{pollutant_name}.csv", index=False)

    df_ridge_final_predictions.to_csv(case_output_dir/ f"ridge_final_test_predictions_{station_name}_{pollutant_name}.csv",index=False)


    save_experiment_metadata(
        output_dir=case_output_dir,
        station_name=station_name,
        pollutant_name=pollutant_name,
        target_col=target_col,
        model_names=model_names,
        feature_sets=feature_sets,
        outer_folds=outer_folds,
    )
    

    return (
        df_fold_metrics_all,
        df_oof_predictions_all,
        df_best_params_all,
        feature_sets,
        outer_folds,
    ) 
# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run nested cross-validation for one pollutant."
    )
    parser.add_argument(
        "--pollutant",
        required=True,
        choices=["NO2", "O3", "PM10", "PM25"],
        help="Pollutant to process.",
    )
    args = parser.parse_args()

    target_col = args.pollutant

    stations = ["LondonMaryleboneRoad","CamdenKerbside","Wandsworth-PutneyHighStreet", "Westminster-OxfordStreet", "Camden-EustonRoad"]
    dir_file = Path("Datasets_to_train_pollutants")
    for station_name in stations:
        
        if station_name == "CamdenKerbside" and (target_col == "O3"):
                continue
        if station_name == "Wandsworth-PutneyHighStreet" and (target_col == "O3" or target_col=="PM25"):
            continue
        if station_name == "Westminster-OxfordStreet" and (target_col == "PM10" or target_col=="PM25" or target_col=="O3"):
            continue
        if station_name == "Camden-EustonRoad" and (target_col == "O3" or target_col == "PM10" or target_col=="PM25"):
            continue                
        

        start = time.time()

        df_case = pd.read_csv(os.path.join(dir_file, f"{station_name}.csv"))
        df_case["date"] = pd.to_datetime(df_case["date"])
        df_case = (df_case.sort_values("date").reset_index(drop=True).dropna(subset=[target_col]).reset_index(drop=True) )
        model_names = ["ridge", "rf", "etr", "lgbm", "xgb"]
    
        print(
            f"Running case for station={station_name}, target={target_col}", flush=True
        )
    
        (
            df_fold_metrics_all,
            df_oof_predictions_all,
            df_best_params_all,
            feature_sets,
            outer_folds,
        ) = run_case(
            df_case=df_case,
            station_name=station_name,
            pollutant_name=target_col,
            target_col=target_col,
            model_names=model_names,
        )
    
        end = time.time()
        print(f"Total time: {(end - start) / 60:.2f} minutes", flush=True)