
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


from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from pyproj import Transformer
from scipy.optimize import least_squares
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# CONFIGURATION
# =========================================================

DATASETS_BY_FOLD_ROOT = Path(
    r"Datasets_to_train_by_fold_pollutants"
)

DISTANCE_MATRIX_FILE = Path(
    "distance_matrix_km_btw_stations.csv"
)

STATION_INFORMATION_FILE = Path(r"\all_stations_more.csv")

OUTPUT_ROOT = Path(
    "interpolation_results_chronological_selection"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_COL = "NO2"

STATIONS = [
    "LondonMaryleboneRoad",
    "Camden-EustonRoad",
    "Westminster-OxfordStreet",
    "CamdenKerbside",
    "Wandsworth-PutneyHighStreet",
]

VARIOGRAM_MODELS = [
    "spherical",
    "exponential",
    "gaussian",
]

DEVELOPMENT_FOLDS = [1, 2, 3, 4]
FINAL_FOLD = 5

# Require all four neighbouring stations for every prediction.
MIN_AUXILIARY_POINTS = 4

# Used only to avoid singular Kriging systems.
USE_PSEUDOINVERSE = True

STATIONS_CONSIDERED = [
    "Camden Kerbside Urban Traffic",
    "London Marylebone Road Urban Traffic",
    "Westminster - Oxford Street Urban Traffic",
    "Camden - Euston Road Urban Traffic",
    "Wandsworth - Putney High Street Urban Traffic",
]

STATION_NAME_TYPE = {
    "CamdenKerbside":
        "Camden Kerbside Urban Traffic",

    "LondonMaryleboneRoad":
        "London Marylebone Road Urban Traffic",

    "Westminster-OxfordStreet":
        "Westminster - Oxford Street Urban Traffic",

    "Camden-EustonRoad":
        "Camden - Euston Road Urban Traffic",

    "Wandsworth-PutneyHighStreet":
        "Wandsworth - Putney High Street Urban Traffic",
}


# =========================================================
# BASIC UTILITIES
# =========================================================

def safe_name(value: object) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
    )


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            mean_squared_error(y_true, y_pred)
        )
    )


def compute_metrics(
    y_true,
    y_pred,
) -> Dict[str, float]:

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    n_valid = int(valid.sum())

    if n_valid < 2:
        return {
            "n_evaluated": n_valid,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
        }

    return {
        "n_evaluated": n_valid,
        "MAE": float(
            mean_absolute_error(
                y_true[valid],
                y_pred[valid],
            )
        ),
        "RMSE": rmse(
            y_true[valid],
            y_pred[valid],
        ),
        "R2": float(
            r2_score(
                y_true[valid],
                y_pred[valid],
            )
        ),
    }


def find_nearest_by_type(
    distance_df: pd.DataFrame,
    stations: Sequence[str],
    suffix: str,
    k: int = 2,
) -> Dict[str, pd.Series]:

    candidates = [
        station
        for station in distance_df.index
        if station.endswith(suffix)
        and station not in stations
    ]

    results = {}

    for station in stations:
        valid_candidates = [
            candidate
            for candidate in candidates
            if candidate != station
        ]

        distances = distance_df.loc[
            station,
            valid_candidates,
        ]

        results[station] = (
            distances
            .sort_values()
            .head(k)
        )

    return results


def get_one_value(
    dataframe: pd.DataFrame,
    filter_column: str,
    filter_value: str,
    value_column: str,
) -> float:

    matched = dataframe.loc[
        dataframe[filter_column] == filter_value,
        value_column,
    ]

    if len(matched) != 1:
        raise ValueError(
            f"Expected exactly one row where "
            f"{filter_column}={filter_value!r}; "
            f"found {len(matched)}."
        )

    return float(matched.iloc[0])


# =========================================================
# FILE LOADING
# =========================================================

def fold_directory(
    pollutant: str,
    station: str,
    outer_fold: int,
) -> Path:

    return (
        DATASETS_BY_FOLD_ROOT
        / pollutant
        / station
        / f"outerfold_{outer_fold}"
    )


def load_outer_fold(
    pollutant: str,
    station: str,
    outer_fold: int,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:

    folder = fold_directory(
        pollutant=pollutant,
        station=station,
        outer_fold=outer_fold,
    )

    x_train_path = (
        folder
        / (
            f"{pollutant}_{station}_"
            f"X_train_outerfold_{outer_fold}.csv"
        )
    )

    y_train_path = (
        folder
        / (
            f"{pollutant}_{station}_"
            f"y_train_outerfold_{outer_fold}.csv"
        )
    )

    x_test_path = (
        folder
        / (
            f"{pollutant}_{station}_"
            f"X_test_outerfold_{outer_fold}.csv"
        )
    )

    y_test_path = (
        folder
        / (
            f"{pollutant}_{station}_"
            f"y_test_outerfold_{outer_fold}.csv"
        )
    )

    required_files = [
        x_train_path,
        y_train_path,
        x_test_path,
        y_test_path,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing fold files:\n"
            + "\n".join(missing_files)
        )

    X_train = pd.read_csv(x_train_path)
    y_train_df = pd.read_csv(y_train_path)

    X_test = pd.read_csv(x_test_path)
    y_test_df = pd.read_csv(y_test_path)

    if pollutant not in y_train_df.columns:
        raise ValueError(
            f"{pollutant} not found in {y_train_path}"
        )

    if pollutant not in y_test_df.columns:
        raise ValueError(
            f"{pollutant} not found in {y_test_path}"
        )

    if len(X_train) != len(y_train_df):
        raise ValueError(
            f"X_train and y_train lengths differ for "
            f"{station}, fold {outer_fold}: "
            f"{len(X_train)} vs {len(y_train_df)}."
        )

    if len(X_test) != len(y_test_df):
        raise ValueError(
            f"X_test and y_test lengths differ for "
            f"{station}, fold {outer_fold}: "
            f"{len(X_test)} vs {len(y_test_df)}."
        )

    return (
        X_train.reset_index(drop=True),
        y_train_df[pollutant].reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_test_df[pollutant].reset_index(drop=True),
    )


# =========================================================
# STATION COORDINATES
# =========================================================

def get_station_coordinates(
    station_name: str,
    distance_df: pd.DataFrame,
    station_information: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[str],
]:
    """
    Return:
        auxiliary_coordinates: shape (4, 2), EPSG:27700
        target_coordinate: shape (2,), EPSG:27700
        auxiliary_names: [bkg0, bkg1, trf0, trf1]
    """

    nearest_background = find_nearest_by_type(
        distance_df=distance_df,
        stations=STATIONS_CONSIDERED,
        suffix="Background",
        k=2,
    )

    nearest_traffic = find_nearest_by_type(
        distance_df=distance_df,
        stations=STATIONS_CONSIDERED,
        suffix="Traffic",
        k=2,
    )

    station_type = STATION_NAME_TYPE[
        station_name
    ]

    background_names = (
        nearest_background[station_type]
        .index
        .tolist()
    )

    traffic_names = (
        nearest_traffic[station_type]
        .index
        .tolist()
    )

    if len(background_names) != 2:
        raise ValueError(
            f"Expected two background stations for "
            f"{station_name}; found {background_names}."
        )

    if len(traffic_names) != 2:
        raise ValueError(
            f"Expected two traffic stations for "
            f"{station_name}; found {traffic_names}."
        )

    clean_background_names = [
        name.split("Urban")[0].strip()
        for name in background_names
    ]

    clean_traffic_names = [
        name.split("Urban")[0].strip()
        for name in traffic_names
    ]

    auxiliary_names = (
        clean_background_names
        + clean_traffic_names
    )

    auxiliary_lons = [
        get_one_value(
            dataframe=station_information,
            filter_column="Site Name",
            filter_value=name,
            value_column="Longitude",
        )
        for name in auxiliary_names
    ]

    auxiliary_lats = [
        get_one_value(
            dataframe=station_information,
            filter_column="Site Name",
            filter_value=name,
            value_column="Latitude",
        )
        for name in auxiliary_names
    ]

    target_lon = get_one_value(
        dataframe=station_information,
        filter_column="Station",
        filter_value=station_name,
        value_column="Longitude",
    )

    target_lat = get_one_value(
        dataframe=station_information,
        filter_column="Station",
        filter_value=station_name,
        value_column="Latitude",
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:27700",
        always_xy=True,
    )

    auxiliary_x, auxiliary_y = (
        transformer.transform(
            auxiliary_lons,
            auxiliary_lats,
        )
    )

    target_x, target_y = transformer.transform(
        target_lon,
        target_lat,
    )

    auxiliary_coordinates = np.column_stack(
        [
            np.asarray(auxiliary_x, dtype=float),
            np.asarray(auxiliary_y, dtype=float),
        ]
    )

    target_coordinate = np.array(
        [target_x, target_y],
        dtype=float,
    )

    return (
        auxiliary_coordinates,
        target_coordinate,
        auxiliary_names,
    )


# =========================================================
# PYKRIGE VARIOGRAM FUNCTIONS
# =========================================================

def spherical_variogram(
    distance: np.ndarray,
    psill: float,
    variogram_range: float,
    nugget: float,
) -> np.ndarray:

    distance = np.asarray(distance, dtype=float)
    ratio = distance / variogram_range

    return np.where(
        distance <= variogram_range,
        nugget
        + psill
        * (
            1.5 * ratio
            - 0.5 * ratio**3
        ),
        nugget + psill,
    )


def exponential_variogram(
    distance: np.ndarray,
    psill: float,
    variogram_range: float,
    nugget: float,
) -> np.ndarray:

    distance = np.asarray(distance, dtype=float)

    return (
        nugget
        + psill
        * (
            1.0
            - np.exp(
                -distance
                / (variogram_range / 3.0)
            )
        )
    )


def gaussian_variogram(
    distance: np.ndarray,
    psill: float,
    variogram_range: float,
    nugget: float,
) -> np.ndarray:

    distance = np.asarray(distance, dtype=float)

    return (
        nugget
        + psill
        * (
            1.0
            - np.exp(
                -(distance**2)
                / (
                    (4.0 * variogram_range / 7.0)
                    ** 2
                )
            )
        )
    )


VARIOGRAM_FUNCTIONS = {
    "spherical": spherical_variogram,
    "exponential": exponential_variogram,
    "gaussian": gaussian_variogram,
}


# =========================================================
# TRAINING DATA FOR THE EMPIRICAL VARIOGRAM
# =========================================================

def build_training_spatial_values(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pollutant: str,
) -> pd.DataFrame:

    auxiliary_columns = [
        f"{pollutant}_bkg0",
        f"{pollutant}_bkg1",
        f"{pollutant}_trf0",
        f"{pollutant}_trf1",
    ]

    missing_columns = [
        column
        for column in auxiliary_columns
        if column not in X_train.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing auxiliary columns: "
            f"{missing_columns}"
        )

    spatial_values = pd.DataFrame(
        {
            "target": pd.to_numeric(
                y_train,
                errors="coerce",
            ),

            "bkg0": pd.to_numeric(
                X_train[auxiliary_columns[0]],
                errors="coerce",
            ),

            "bkg1": pd.to_numeric(
                X_train[auxiliary_columns[1]],
                errors="coerce",
            ),

            "trf0": pd.to_numeric(
                X_train[auxiliary_columns[2]],
                errors="coerce",
            ),

            "trf1": pd.to_numeric(
                X_train[auxiliary_columns[3]],
                errors="coerce",
            ),
        }
    )

    return spatial_values.reset_index(drop=True)


# =========================================================
# EMPIRICAL VARIOGRAM
# =========================================================

def estimate_empirical_variogram(
    training_values: pd.DataFrame,
    coordinates: np.ndarray,
) -> pd.DataFrame:
    """
    Estimate one semivariance for every pair of monitoring stations.

    For station pair i,j:

        gamma_ij =
            median_t[0.5 * (z_i(t) - z_j(t))²]

    The median is used to reduce sensitivity to extreme hourly
    concentration differences.
    """

    if training_values.shape[1] != len(coordinates):
        raise ValueError(
            "The number of station columns does not match "
            "the number of coordinates."
        )

    records = []

    for i, j in combinations(
        range(training_values.shape[1]),
        2,
    ):
        station_i = training_values.columns[i]
        station_j = training_values.columns[j]

        values_i = pd.to_numeric(
            training_values.iloc[:, i],
            errors="coerce",
        )

        values_j = pd.to_numeric(
            training_values.iloc[:, j],
            errors="coerce",
        )

        valid = (
            values_i.notna()
            & values_j.notna()
        )

        n_pairs = int(valid.sum())

        if n_pairs == 0:
            continue

        differences = (
            values_i.loc[valid].to_numpy(float)
            - values_j.loc[valid].to_numpy(float)
        )

        hourly_semivariances = (
            0.5 * differences**2
        )

        distance_m = float(
            np.linalg.norm(
                coordinates[i]
                - coordinates[j]
            )
        )

        records.append(
            {
                "station_i": station_i,
                "station_j": station_j,
                "distance_m": distance_m,
                "distance_km": distance_m / 1000,
                "semivariance": float(
                    np.median(hourly_semivariances)
                ),
                "mean_semivariance": float(
                    np.mean(hourly_semivariances)
                ),
                "n_time_pairs": n_pairs,
            }
        )

    empirical = pd.DataFrame(records)

    if len(empirical) < 4:
        raise ValueError(
            "Too few valid station pairs to fit a variogram."
        )

    return (
        empirical
        .sort_values("distance_m")
        .reset_index(drop=True)
    )


# =========================================================
# PARAMETRIC VARIOGRAM FITTING
# =========================================================

def fit_variogram_parameters(
    empirical_variogram: pd.DataFrame,
    model_name: str,
) -> Dict[str, float]:

    if model_name not in VARIOGRAM_FUNCTIONS:
        raise ValueError(
            f"Unsupported variogram model: {model_name}"
        )

    distances = empirical_variogram[
        "distance_m"
    ].to_numpy(float)

    observed_semivariance = empirical_variogram[
        "semivariance"
    ].to_numpy(float)

    pair_counts = empirical_variogram[
        "n_time_pairs"
    ].to_numpy(float)

    valid = (
        np.isfinite(distances)
        & np.isfinite(observed_semivariance)
        & (distances > 0)
    )

    distances = distances[valid]
    observed_semivariance = (
        observed_semivariance[valid]
    )
    pair_counts = pair_counts[valid]

    if len(distances) < 4:
        raise ValueError(
            "Too few valid empirical variogram points."
        )

    maximum_distance = float(
        distances.max()
    )

    minimum_distance = float(
        distances.min()
    )

    maximum_semivariance = max(
        float(observed_semivariance.max()),
        1e-8,
    )

    initial_nugget = max(
        float(
            np.percentile(
                observed_semivariance,
                5,
            )
        ),
        0.0,
    )

    initial_psill = max(
        float(
            np.percentile(
                observed_semivariance,
                75,
            )
        )
        - initial_nugget,
        1e-8,
    )

    initial_range = float(
        np.median(distances)
    )

    lower_bounds = np.array(
        [
            1e-10,
            minimum_distance * 0.05,
            0.0,
        ],
        dtype=float,
    )

    upper_bounds = np.array(
        [
            maximum_semivariance * 20,
            maximum_distance * 10,
            maximum_semivariance * 10,
        ],
        dtype=float,
    )

    initial_parameters = np.array(
        [
            initial_psill,
            initial_range,
            initial_nugget,
        ],
        dtype=float,
    )

    initial_parameters = np.clip(
        initial_parameters,
        lower_bounds + 1e-12,
        upper_bounds - 1e-12,
    )

    variogram_function = VARIOGRAM_FUNCTIONS[
        model_name
    ]

    weights = np.sqrt(
        pair_counts / pair_counts.max()
    )

    def weighted_residuals(parameters):
        psill, variogram_range, nugget = (
            parameters
        )

        fitted = variogram_function(
            distances,
            psill,
            variogram_range,
            nugget,
        )

        return weights * (
            fitted - observed_semivariance
        )

    optimisation = least_squares(
        weighted_residuals,
        x0=initial_parameters,
        bounds=(
            lower_bounds,
            upper_bounds,
        ),
        method="trf",
    )

    psill, variogram_range, nugget = (
        optimisation.x
    )

    fitted_semivariance = variogram_function(
        distances,
        psill,
        variogram_range,
        nugget,
    )

    fit_rmse = float(
        np.sqrt(
            np.mean(
                (
                    fitted_semivariance
                    - observed_semivariance
                ) ** 2
            )
        )
    )

    return {
        "variogram_model": model_name,
        "psill": float(psill),
        "sill": float(psill + nugget),
        "range_m": float(variogram_range),
        "range_km": float(
            variogram_range / 1000
        ),
        "nugget": float(nugget),
        "variogram_fit_RMSE": fit_rmse,
        "optimisation_success": bool(
            optimisation.success
        ),
        "optimisation_message": str(
            optimisation.message
        ),
    }


# =========================================================
# KRIGING PREDICTION
# =========================================================

def predict_one_timestamp(
    auxiliary_coordinates: np.ndarray,
    auxiliary_values: np.ndarray,
    target_coordinate: np.ndarray,
    variogram_model: str,
    parameters: Dict[str, float],
) -> Tuple[float, float, int]:

    auxiliary_values = np.asarray(
        auxiliary_values,
        dtype=float,
    )

    valid = np.isfinite(auxiliary_values)
    n_valid = int(valid.sum())

    if n_valid < MIN_AUXILIARY_POINTS:
        return np.nan, np.nan, n_valid

    coordinates_valid = (
        auxiliary_coordinates[valid]
    )

    values_valid = auxiliary_values[valid]

    kriging = OrdinaryKriging(
        coordinates_valid[:, 0],
        coordinates_valid[:, 1],
        values_valid,
        variogram_model=variogram_model,
        variogram_parameters={
            "psill": parameters["psill"],
            "range": parameters["range_m"],
            "nugget": parameters["nugget"],
        },
        coordinates_type="euclidean",
        exact_values=True,
        pseudo_inv=USE_PSEUDOINVERSE,
        pseudo_inv_type="pinv",
        verbose=False,
        enable_plotting=False,
    )

    prediction, variance = kriging.execute(
        "points",
        np.array([target_coordinate[0]]),
        np.array([target_coordinate[1]]),
    )

    prediction_value = float(
        np.asarray(prediction).ravel()[0]
    )

    variance_value = float(
        np.asarray(variance).ravel()[0]
    )

    return (
        prediction_value,
        variance_value,
        n_valid,
    )


def predict_fold_test_set(
    X_test: pd.DataFrame,
    pollutant: str,
    auxiliary_coordinates: np.ndarray,
    target_coordinate: np.ndarray,
    variogram_model: str,
    parameters: Dict[str, float],
) -> pd.DataFrame:

    auxiliary_columns = [
        f"{pollutant}_bkg0",
        f"{pollutant}_bkg1",
        f"{pollutant}_trf0",
        f"{pollutant}_trf1",
    ]

    missing_columns = [
        column
        for column in auxiliary_columns
        if column not in X_test.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing test columns: {missing_columns}"
        )

    auxiliary_values = (
        X_test[auxiliary_columns]
        .apply(pd.to_numeric, errors="coerce")
    )

    predictions = []
    variances = []
    n_auxiliary_points = []

    for _, row in auxiliary_values.iterrows():

        prediction, variance, n_valid = (
            predict_one_timestamp(
                auxiliary_coordinates=(
                    auxiliary_coordinates
                ),
                auxiliary_values=(
                    row.to_numpy(float)
                ),
                target_coordinate=(
                    target_coordinate
                ),
                variogram_model=variogram_model,
                parameters=parameters,
            )
        )

        predictions.append(prediction)
        variances.append(variance)
        n_auxiliary_points.append(n_valid)

    return pd.DataFrame(
        {
            "y_pred": predictions,
            "kriging_variance": variances,
            "n_aux_points": n_auxiliary_points,
        }
    )


# =========================================================
# FIT AND EVALUATE ONE OUTER FOLD
# =========================================================

def evaluate_variogram_on_fold(
    station_name: str,
    pollutant: str,
    outer_fold: int,
    variogram_model: str,
    auxiliary_coordinates: np.ndarray,
    target_coordinate: np.ndarray,
) -> Tuple[
    Dict[str, object],
    pd.DataFrame,
    Dict[str, object],
    pd.DataFrame,
]:

    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = load_outer_fold(
        pollutant=pollutant,
        station=station_name,
        outer_fold=outer_fold,
    )

    training_values = (
        build_training_spatial_values(
            X_train=X_train,
            y_train=y_train,
            pollutant=pollutant,
        )
    )

    all_coordinates = np.vstack(
        [
            target_coordinate,
            auxiliary_coordinates,
        ]
    )

    empirical_variogram = (
        estimate_empirical_variogram(
            training_values=training_values,
            coordinates=all_coordinates,
        )
    )

    parameters = fit_variogram_parameters(
        empirical_variogram=empirical_variogram,
        model_name=variogram_model,
    )

    test_results = predict_fold_test_set(
        X_test=X_test,
        pollutant=pollutant,
        auxiliary_coordinates=(
            auxiliary_coordinates
        ),
        target_coordinate=target_coordinate,
        variogram_model=variogram_model,
        parameters=parameters,
    )

    y_true = pd.to_numeric(
        y_test,
        errors="coerce",
    ).to_numpy(float)

    y_pred = test_results[
        "y_pred"
    ].to_numpy(float)

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    prediction_frame = pd.DataFrame(
        {
            "station": station_name,
            "pollutant": pollutant,
            "scenario": "Interpolation",
            "model_name": "OK",
            "variogram_model": variogram_model,
            "outer_fold": outer_fold,
            "split_role": (
                "development_oof"
                if outer_fold < FINAL_FOLD
                else "final_test_interpolation"
            ),
            "row_id_original": (
                X_test["row_id_original"].to_numpy()
                if "row_id_original" in X_test.columns
                else np.arange(len(X_test))
            ),
            "date": (
                X_test["date"].to_numpy()
                if "date" in X_test.columns
                else pd.NaT
            ),
            "y_true": y_true,
            "y_pred": y_pred,
            "kriging_variance": (
                test_results[
                    "kriging_variance"
                ].to_numpy(float)
            ),
            "n_aux_points": (
                test_results[
                    "n_aux_points"
                ].to_numpy(int)
            ),
        }
    )

    metric_record = {
        "station": station_name,
        "pollutant": pollutant,
        "scenario": "Interpolation",
        "model_name": "OK",
        "variogram_model": variogram_model,
        "outer_fold": outer_fold,
        "n_train": len(X_train),
        "n_test_total": len(X_test),
        "n_test_evaluated": (
            metrics["n_evaluated"]
        ),
        "n_test_excluded": (
            len(X_test)
            - metrics["n_evaluated"]
        ),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
    }

    parameter_record = {
        "station": station_name,
        "pollutant": pollutant,
        "variogram_model": variogram_model,
        "outer_fold": outer_fold,
        "training_period": (
            "development"
            if outer_fold < FINAL_FOLD
            else "final_fold_train"
        ),
        **parameters,
    }

    empirical_variogram = (
        empirical_variogram.copy()
    )

    empirical_variogram.insert(
        0,
        "station",
        station_name,
    )

    empirical_variogram.insert(
        1,
        "pollutant",
        pollutant,
    )

    empirical_variogram.insert(
        2,
        "outer_fold",
        outer_fold,
    )

    return (
        metric_record,
        prediction_frame,
        parameter_record,
        empirical_variogram,
    )


# =========================================================
# TEMPORAL STABILITY DIAGNOSTIC
# =========================================================

def variogram_temporal_sensitivity(
    station_name: str,
    pollutant: str,
    selected_model: str,
    auxiliary_coordinates: np.ndarray,
    target_coordinate: np.ndarray,
) -> pd.DataFrame:
    """
    Fit the selected variogram separately to the first and second
    halves of the fold-5 training period.

    This is a sensitivity diagnostic, not a formal stationarity test.
    """

    X_train, y_train, _, _ = load_outer_fold(
        pollutant=pollutant,
        station=station_name,
        outer_fold=FINAL_FOLD,
    )

    training_values = (
        build_training_spatial_values(
            X_train=X_train,
            y_train=y_train,
            pollutant=pollutant,
        )
    )

    midpoint = len(training_values) // 2

    periods = {
        "first_half": (
            training_values.iloc[:midpoint]
        ),
        "second_half": (
            training_values.iloc[midpoint:]
        ),
        "complete_fold5_train": (
            training_values
        ),
    }

    all_coordinates = np.vstack(
        [
            target_coordinate,
            auxiliary_coordinates,
        ]
    )

    records = []

    for period_name, period_values in periods.items():

        empirical = estimate_empirical_variogram(
            training_values=period_values,
            coordinates=all_coordinates,
        )

        parameters = fit_variogram_parameters(
            empirical_variogram=empirical,
            model_name=selected_model,
        )

        records.append(
            {
                "station": station_name,
                "pollutant": pollutant,
                "variogram_model": (
                    selected_model
                ),
                "training_period": period_name,
                "n_timestamps": len(
                    period_values
                ),
                **parameters,
            }
        )

    return pd.DataFrame(records)


# =========================================================
# COMPLETE PIPELINE FOR ONE STATION
# =========================================================

def run_station_kriging(
    station_name: str,
    pollutant: str,
    distance_df: pd.DataFrame,
    station_information: pd.DataFrame,
) -> None:

    print(
        f"\n{'=' * 70}\n"
        f"Station: {station_name} | "
        f"Pollutant: {pollutant}\n"
        f"{'=' * 70}",
        flush=True,
    )

    station_output_dir = (
        OUTPUT_ROOT
        / f"{safe_name(station_name)}_{pollutant}"
    )

    station_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        auxiliary_coordinates,
        target_coordinate,
        auxiliary_names,
    ) = get_station_coordinates(
        station_name=station_name,
        distance_df=distance_df,
        station_information=station_information,
    )

    coordinate_records = pd.DataFrame(
        {
            "role": [
                "target",
                "bkg0",
                "bkg1",
                "trf0",
                "trf1",
            ],
            "station_name": [
                station_name,
                *auxiliary_names,
            ],
            "x_epsg27700_m": [
                target_coordinate[0],
                *auxiliary_coordinates[:, 0],
            ],
            "y_epsg27700_m": [
                target_coordinate[1],
                *auxiliary_coordinates[:, 1],
            ],
        }
    )

    coordinate_records.to_csv(
        station_output_dir
        / "kriging_station_coordinates.csv",
        index=False,
    )

    development_metrics = []
    development_predictions = []
    development_parameters = []
    empirical_variograms = []

    # -----------------------------------------------------
    # Development folds 1–4
    # -----------------------------------------------------

    for outer_fold in DEVELOPMENT_FOLDS:

        for variogram_model in VARIOGRAM_MODELS:

            print(
                f"Development fold {outer_fold} | "
                f"{variogram_model}",
                flush=True,
            )

            (
                metric_record,
                predictions,
                parameter_record,
                empirical_variogram,
            ) = evaluate_variogram_on_fold(
                station_name=station_name,
                pollutant=pollutant,
                outer_fold=outer_fold,
                variogram_model=(
                    variogram_model
                ),
                auxiliary_coordinates=(
                    auxiliary_coordinates
                ),
                target_coordinate=(
                    target_coordinate
                ),
            )

            development_metrics.append(
                metric_record
            )

            development_predictions.append(
                predictions
            )

            development_parameters.append(
                parameter_record
            )

            # Empirical variogram is identical for the
            # three candidate models within the same fold.
            if variogram_model == VARIOGRAM_MODELS[0]:
                empirical_variograms.append(
                    empirical_variogram
                )

            print(
                f"  RMSE={metric_record['RMSE']:.4f} | "
                f"MAE={metric_record['MAE']:.4f} | "
                f"R2={metric_record['R2']:.4f}",
                flush=True,
            )

    development_metrics_df = pd.DataFrame(
        development_metrics
    )

    development_predictions_df = pd.concat(
        development_predictions,
        ignore_index=True,
    )

    development_parameters_df = pd.DataFrame(
        development_parameters
    )

    empirical_variograms_df = pd.concat(
        empirical_variograms,
        ignore_index=True,
    )

    # -----------------------------------------------------
    # Select variogram by mean RMSE across folds 1–4
    # -----------------------------------------------------

    selection_summary = (
        development_metrics_df
        .groupby(
            "variogram_model",
            as_index=False,
        )
        .agg(
            mean_RMSE=("RMSE", "mean"),
            std_RMSE=("RMSE", "std"),
            mean_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            mean_R2=("R2", "mean"),
            std_R2=("R2", "std"),
            n_development_folds=(
                "outer_fold",
                "nunique",
            ),
        )
        .sort_values(
            [
                "mean_RMSE",
                "variogram_model",
            ]
        )
        .reset_index(drop=True)
    )

    selected_variogram_model = str(
        selection_summary.loc[
            0,
            "variogram_model",
        ]
    )

    selection_summary["selected"] = (
        selection_summary["variogram_model"]
        == selected_variogram_model
    )

    selection_summary.insert(
        0,
        "station",
        station_name,
    )

    selection_summary.insert(
        1,
        "pollutant",
        pollutant,
    )

    print(
        f"\nSelected variogram: "
        f"{selected_variogram_model}",
        flush=True,
    )

    # -----------------------------------------------------
    # Final fold 5
    # Refit selected variogram on fold-5 train and evaluate
    # once on fold-5 test.
    # -----------------------------------------------------

    (
        final_metric_record,
        final_predictions,
        final_parameter_record,
        final_empirical_variogram,
    ) = evaluate_variogram_on_fold(
        station_name=station_name,
        pollutant=pollutant,
        outer_fold=FINAL_FOLD,
        variogram_model=selected_variogram_model,
        auxiliary_coordinates=(
            auxiliary_coordinates
        ),
        target_coordinate=target_coordinate,
    )

    selection_row = (
        selection_summary.loc[
            selection_summary["selected"]
        ]
        .iloc[0]
    )

    final_metric_record.update(
        {
            "selected_using_folds":
                "1,2,3,4",

            "selection_metric":
                "RMSE",

            "development_mean_RMSE":
                float(
                    selection_row["mean_RMSE"]
                ),

            "development_std_RMSE":
                float(
                    selection_row["std_RMSE"]
                ),
        }
    )

    final_metrics_df = pd.DataFrame(
        [final_metric_record]
    )

    final_parameters_df = pd.DataFrame(
        [final_parameter_record]
    )

    # -----------------------------------------------------
    # Temporal sensitivity of the selected variogram
    # -----------------------------------------------------

    temporal_sensitivity_df = (
        variogram_temporal_sensitivity(
            station_name=station_name,
            pollutant=pollutant,
            selected_model=(
                selected_variogram_model
            ),
            auxiliary_coordinates=(
                auxiliary_coordinates
            ),
            target_coordinate=(
                target_coordinate
            ),
        )
    )

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------

    development_metrics_df.to_csv(
        station_output_dir
        / "kriging_development_fold_metrics.csv",
        index=False,
    )

    development_predictions_df.to_csv(
        station_output_dir
        / "kriging_development_oof_predictions.csv",
        index=False,
    )

    development_parameters_df.to_csv(
        station_output_dir
        / "kriging_development_variogram_parameters.csv",
        index=False,
    )

    empirical_variograms_df.to_csv(
        station_output_dir
        / "kriging_development_empirical_variograms.csv",
        index=False,
    )

    selection_summary.to_csv(
        station_output_dir
        / "kriging_variogram_selection.csv",
        index=False,
    )

    final_predictions.to_csv(
        station_output_dir
        / (
            f"final_test_predictions_OK_"
            f"{station_name}_{pollutant}.csv"
        ),
        index=False,
    )

    final_metrics_df.to_csv(
        station_output_dir
        / (
            f"final_test_metrics_OK_"
            f"{station_name}_{pollutant}.csv"
        ),
        index=False,
    )

    final_parameters_df.to_csv(
        station_output_dir
        / (
            f"final_variogram_parameters_"
            f"{station_name}_{pollutant}.csv"
        ),
        index=False,
    )

    final_empirical_variogram.to_csv(
        station_output_dir
        / (
            f"final_empirical_variogram_"
            f"{station_name}_{pollutant}.csv"
        ),
        index=False,
    )

    temporal_sensitivity_df.to_csv(
        station_output_dir
        / (
            f"variogram_temporal_sensitivity_"
            f"{station_name}_{pollutant}.csv"
        ),
        index=False,
    )

    print(
        "\nFinal test results:"
    )

    print(
        final_metrics_df.to_string(
            index=False
        )
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    distance_df = pd.read_csv(
        DISTANCE_MATRIX_FILE
    )

    if "Sitename_type" not in distance_df.columns:
        raise ValueError(
            "The distance matrix must contain "
            "'Sitename_type'."
        )

    distance_df = distance_df.set_index(
        "Sitename_type"
    )

    station_information = pd.read_csv(
        STATION_INFORMATION_FILE
    )

    required_station_columns = {
        "Station",
        "Site Name",
        "Longitude",
        "Latitude",
    }

    missing_station_columns = (
        required_station_columns.difference(
            station_information.columns
        )
    )

    if missing_station_columns:
        raise ValueError(
            "Missing station-information columns: "
            f"{sorted(missing_station_columns)}"
        )

    completed = []
    failed = []

    for station_name in STATIONS:

        try:
            run_station_kriging(
                station_name=station_name,
                pollutant=TARGET_COL,
                distance_df=distance_df,
                station_information=(
                    station_information
                ),
            )

            completed.append(station_name)

        except Exception as error:
            failed.append(
                {
                    "station": station_name,
                    "pollutant": TARGET_COL,
                    "error_type": (
                        type(error).__name__
                    ),
                    "error": str(error),
                }
            )

            print(
                f"\nFAILED: {station_name} | "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
    

    print(
        f"\nCompleted stations: {completed}"
    )

    if failed:
        print(
            f"Failed cases:\n"
            f"{pd.DataFrame(failed).to_string(index=False)}"
        )


if __name__ == "__main__":
    main()

