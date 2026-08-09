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


import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

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

def interpolates_variables(
    df: pd.DataFrame, 
    columns_to_interpolate: list[str],
    datetime_column: str = "date",
    max_gap_hours: int = 12,
    return_interpolated=False
) -> pd.DataFrame:
    """
    Temporally interpolates internal gaps containing up to
    max_consecutive_missing consecutive observations.

    The interpolation uses the datetime spacing, but the limit refers
    to the number of consecutive missing rows rather than an exact
    elapsed duration.
    """

    result = df.copy()
    result[datetime_column] = pd.to_datetime(result[datetime_column])
    result = result.sort_values(datetime_column)
    original_index = result.index
    result = result.set_index(datetime_column)

    missing_columns = [column for column in columns_to_interpolate if column not in result.columns]
    if missing_columns:
        raise ValueError(f"Columns not found: {missing_columns}")

    before_missing = result[columns_to_interpolate].isna()
    result[columns_to_interpolate] = result[columns_to_interpolate].interpolate(method="time", limit=max_gap_hours, limit_area="inside")
    after_missing = result[columns_to_interpolate].isna()
    interpolated_mask = before_missing & ~after_missing


    result = result.reset_index()
    result.index = original_index
    interpolated_mask = interpolated_mask.reset_index(drop=True)
    interpolated_mask.index = original_index
    if return_interpolated:
        return result, interpolated_mask

    
    return result


def safe_name(text):
    return str(text).replace(" ", "_").replace("/", "_").replace("-", "_")

N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3
stations = ["Wandsworth-PutneyHighStreet","LondonMaryleboneRoad","CamdenKerbside", "Westminster-OxfordStreet", "Camden-EustonRoad"]
dir_file = Path("Datasets_to_train")
for target_col in ["NO2", "O3", "PM10", "PM25"]:
    for station_name in stations:
        
        if station_name == "CamdenKerbside" and (target_col == "O3"):
                continue
        if station_name == "Wandsworth-PutneyHighStreet" and (target_col == "O3" or target_col=="PM25"):
            continue
        if station_name == "Westminster-OxfordStreet" and (target_col == "PM10" or target_col=="PM25" or target_col=="O3"):
            continue
        if station_name == "Camden-EustonRoad" and (target_col == "O3" or target_col == "PM10" or target_col=="PM25"):
            continue                
        
        print( f"Running case for station={station_name}, target={target_col}", flush=True  )
        df_case = pd.read_csv(os.path.join(dir_file, f"{station_name}.csv"))
        df_case["date"] = pd.to_datetime(df_case["date"])
        df_case = df_case.sort_values("date").reset_index(drop=True)
        n_missing_target = df_case[target_col].isna().sum()
        df_case = df_case.dropna(subset=[target_col]).reset_index(drop=True)
        n_after_target_removal = len(df_case)           

        outer_folds = make_outer_folds(df_case, n_splits=N_OUTER_SPLITS)
        
        bkg0_vars = [col for col in df_case.columns if "bkg0" in col]
        bkg1_vars = [col for col in df_case.columns if "bkg1" in col]
        trf0_vars = [col for col in df_case.columns if "trf0" in col]
        trf1_vars = [col for col in df_case.columns if "trf1" in col]
        TRAFFIC_VARS = [ "traffic_level", "currenttraveltime"]
        TEMPORAL_VARS = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos"]
        MET_VARS = [ "temp_ow", "pressure_ow", "humidity_ow", "clouds_ow", "wind_speed_ow","wdr_sin", "wdr_cos","dew_point_ow"]
        df_case["row_id_original"] = np.arange(len(df_case))
        feature_cols = ["date", "row_id_original"] + MET_VARS + TEMPORAL_VARS + TRAFFIC_VARS + bkg0_vars + bkg1_vars + trf0_vars + trf1_vars

        X_all = df_case[feature_cols].copy()
        y_all = df_case[target_col].copy()

        predictor_cols = [c for c in feature_cols if c not in ["date", "row_id_original"]]
        n_missing_predictor_values = X_all[predictor_cols].isna().sum().sum()
        n_rows_with_missing_predictors = X_all[predictor_cols].isna().any(axis=1).sum()

        interpolated_unique = set()

        for outer_fold, (train_idx, test_idx) in enumerate(outer_folds, start=1):
            print(f"Running nested CV fold {outer_fold}/{len(outer_folds)}", flush=True)

            X_train_outer = X_all.iloc[train_idx].copy()
            y_train_outer = y_all.iloc[train_idx].copy()
            X_test_outer = X_all.iloc[test_idx].copy()
            y_test_outer = y_all.iloc[test_idx].copy()


            outer_dir = Path(f"Datasets_to_train_by_fold_pollutants/{target_col}/{station_name}/outerfold_{outer_fold}" )
            outer_dir.mkdir(parents=True, exist_ok=True)

            inner_dir = outer_dir / "innerfold"
            inner_dir.mkdir(parents=True, exist_ok=True)
            
            inner_folds = make_inner_folds(X_train_outer, n_splits=N_INNER_SPLITS)

            for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(inner_folds, start=1):
                X_tr = X_train_outer.iloc[inner_train_idx].copy()
                y_tr = y_train_outer.iloc[inner_train_idx].copy()
                X_val = X_train_outer.iloc[inner_valid_idx].copy()
                y_val = y_train_outer.iloc[inner_valid_idx].copy()

                # Missing values imputation
                X_tr = interpolates_variables(X_tr, columns_to_interpolate=X_tr.columns.drop(["date", "row_id_original"]), datetime_column="date", max_gap_hours=4)
                X_val = interpolates_variables(X_val, columns_to_interpolate=X_val.columns.drop(["date", "row_id_original"]), datetime_column="date", max_gap_hours=4)
                
                X_tr.to_csv(os.path.join(inner_dir, f"{target_col}_{station_name}_X_tr_outerfold_{outer_fold}_innerfold_{inner_fold}.csv"), index=False)
                y_tr.to_csv(os.path.join(inner_dir, f"{target_col}_{station_name}_y_tr_outerfold_{outer_fold}_innerfold_{inner_fold}.csv"), index=False)
                X_val.to_csv(os.path.join(inner_dir, f"{target_col}_{station_name}_X_val_outerfold_{outer_fold}_innerfold_{inner_fold}.csv"), index=False)
                y_val.to_csv(os.path.join(inner_dir, f"{target_col}_{station_name}_y_val_outerfold_{outer_fold}_innerfold_{inner_fold}.csv"), index=False)
                
            # Missing values imputation
            X_train_outer, mask_train = interpolates_variables(X_train_outer, columns_to_interpolate=X_train_outer.columns.drop(["date", "row_id_original"]), datetime_column="date", max_gap_hours=4, return_interpolated=True)
            X_test_outer, mask_test  = interpolates_variables(X_test_outer, columns_to_interpolate=X_test_outer.columns.drop(["date", "row_id_original"]), datetime_column="date", max_gap_hours=4, return_interpolated=True)
            
            for col in mask_train.columns:
                for idx in mask_train.index[mask_train[col]]:
                    interpolated_unique.add((int(X_train_outer.loc[idx, "row_id_original"]), col))
            
            for col in mask_test.columns:
                for idx in mask_test.index[mask_test[col]]:
                    interpolated_unique.add((int(X_test_outer.loc[idx, "row_id_original"]), col))
            assert len(X_tr) == len(y_tr)
            assert len(X_val) == len(y_val)
            assert len(X_train_outer) == len(y_train_outer)
            assert len(X_test_outer) == len(y_test_outer)

            X_train_outer.to_csv(os.path.join(outer_dir, f"{target_col}_{station_name}_X_train_outerfold_{outer_fold}.csv"), index=False)
            y_train_outer.to_csv(os.path.join(outer_dir, f"{target_col}_{station_name}_y_train_outerfold_{outer_fold}.csv"), index=False)
            X_test_outer.to_csv(os.path.join(outer_dir, f"{target_col}_{station_name}_X_test_outerfold_{outer_fold}.csv"), index=False)
            y_test_outer.to_csv(os.path.join(outer_dir, f"{target_col}_{station_name}_y_test_outerfold_{outer_fold}.csv"), index=False)
            
                        
        n_interpolated_predictor_values = len(interpolated_unique)
        summary_file = Path("preprocessing_data_counts.csv")
        df_counts = pd.read_csv(summary_file)
        station_map = {"LondonMaryleboneRoad": "London Marylebone Road", "CamdenKerbside": "Camden Kerbside", "Wandsworth-PutneyHighStreet": "Wandsworth - Putney High Street", "Westminster-OxfordStreet": "Westminster - Oxford Street", "Camden-EustonRoad": "Camden - Euston Road"}
        station_table = station_map.get(station_name, station_name)
        mask = (df_counts["station"] == station_table) & (df_counts["pollutant"] == target_col)
        
                    
        df_counts.loc[mask, "n_missing_target"] = n_missing_target
        df_counts.loc[mask, "n_after_target_removal"] = n_after_target_removal
        df_counts.loc[mask, "n_missing_predictor_values"] = n_missing_predictor_values
        df_counts.loc[mask, "n_rows_with_missing_predictors"] = n_rows_with_missing_predictors
        df_counts.loc[mask, "n_interpolated_predictor_values"] = n_interpolated_predictor_values
        df_counts.to_csv(summary_file, index=False)