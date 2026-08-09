from __future__ import annotations
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path(
    "nested_cv_results_pollutants"
)

OUTPUT_DIR = Path(
    "Tables_ML_vs_Ridge"
)


PROJECT_DIR = Path(
    "."
)

RESULTS_DIR = (
    PROJECT_DIR
    / "Datasets_to_train"
    / "nested_cv_results_pollutants"
)

OUTPUT_DIR = Path("Tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


POLLUTANTS = [
    "NO2",
    "O3",
    "PM10",
    "PM25",
]

STATIONS = [
    "LondonMaryleboneRoad",
    "CamdenKerbside",
    "Wandsworth-PutneyHighStreet",
    "Westminster-OxfordStreet",
    "Camden-EustonRoad",
]


SCENARIO_ORDER = [
    "baseline_tm",
    "traffic_tm",
    "nearest_bg_no_traffic",
    "nearest_bg_plus_traffic",
    "multi_station_no_traffic",
    "multi_station_plus_traffic",
]


SCENARIO_LABELS = {
    "baseline_tm": "Scen1-noT",
    "traffic_tm": "Scen1",
    "nearest_bg_no_traffic": "Scen2-noT",
    "nearest_bg_plus_traffic": "Scen2",
    "multi_station_no_traffic": "Scen3-noT",
    "multi_station_plus_traffic": "Scen3",
}


STATION_LABELS = {
    "LondonMaryleboneRoad": "Marylebone Road",
    "CamdenKerbside": "Camden Kerbside",
    "Wandsworth-PutneyHighStreet": "Putney High Street",
    "Westminster-OxfordStreet": "Oxford Street",
    "Camden-EustonRoad": "Euston Road",
}


MODEL_LABELS = {
    "rf": "RF",
    "etr": "ETR",
    "lgbm": "LGBM",
    "xgb": "XGB",
    "ridge": "Ridge",
}


DECIMALS = {
    "RMSE": 2,
    "MAE": 2,
    "R2": 3,
}


def safe_name(text: str) -> str:
    """Replicates the folder naming used by the training script."""
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def case_is_available(
    station: str,
    pollutant: str,
) -> bool:
    """
    Reproduces the station-pollutant exclusions in the
    training script.
    """

    unavailable_cases = {
        ("CamdenKerbside", "O3"),

        ("Wandsworth-PutneyHighStreet", "O3"),
        ("Wandsworth-PutneyHighStreet", "PM25"),

        ("Westminster-OxfordStreet", "O3"),
        ("Westminster-OxfordStreet", "PM10"),
        ("Westminster-OxfordStreet", "PM25"),

        ("Camden-EustonRoad", "O3"),
        ("Camden-EustonRoad", "PM10"),
        ("Camden-EustonRoad", "PM25"),
    }

    return (station, pollutant) not in unavailable_cases


def get_case_directory(
    station: str,
    pollutant: str,
) -> Path:
    return (
        RESULTS_DIR
        / f"{safe_name(station)}_{safe_name(pollutant)}"
    )


def validate_metric_file(
    df: pd.DataFrame,
    file_path: Path,
    expected_model_type: str,
) -> None:
    """Checks that the final metric file has the expected structure."""

    required_columns = {
        "station",
        "pollutant",
        "scenario",
        "selected_model_name",
        "Final_test_RMSE",
        "Final_test_MAE",
        "Final_test_R2",
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"{file_path} is missing columns: {sorted(missing)}"
        )

    duplicated = df.duplicated(
        subset=[
            "station",
            "pollutant",
            "scenario",
        ],
        keep=False,
    )

    if duplicated.any():
        duplicate_rows = df.loc[
            duplicated,
            ["station", "pollutant", "scenario"],
        ]

        raise ValueError(
            f"Duplicated station-pollutant-scenario rows in "
            f"{file_path}:\n{duplicate_rows}"
        )

    metric_columns = [
        "Final_test_RMSE",
        "Final_test_MAE",
        "Final_test_R2",
    ]

    for column in metric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[metric_columns].isna().any().any():
        missing_counts = (
            df[metric_columns]
            .isna()
            .sum()
        )

        raise ValueError(
            f"Invalid metric values in {file_path}:\n"
            f"{missing_counts[missing_counts > 0]}"
        )

    if expected_model_type == "ridge":
        non_ridge = (
            df["selected_model_name"]
            .astype(str)
            .str.lower()
            .ne("ridge")
        )

        if non_ridge.any():
            raise ValueError(
                f"The Ridge file contains non-Ridge model names: "
                f"{df.loc[non_ridge, 'selected_model_name'].unique()}"
            )



def load_case_metrics(
    station: str,
    pollutant: str,
) -> pd.DataFrame:

    case_dir = get_case_directory(
        station=station,
        pollutant=pollutant,
    )

    ml_path = (
        case_dir
        / f"final_test_metrics_{station}_{pollutant}.csv"
    )

    ridge_path = (
        case_dir
        / f"ridge_final_test_metrics_{station}_{pollutant}.csv"
    )

    if not ml_path.is_file():
        raise FileNotFoundError(
            f"Best-ML metric file not found:\n{ml_path}"
        )

    if not ridge_path.is_file():
        raise FileNotFoundError(
            f"Ridge metric file not found:\n{ridge_path}"
        )

    ml = pd.read_csv(ml_path)
    ridge = pd.read_csv(ridge_path)

    validate_metric_file(
        df=ml,
        file_path=ml_path,
        expected_model_type="ml",
    )

    validate_metric_file(
        df=ridge,
        file_path=ridge_path,
        expected_model_type="ridge",
    )

    ml = ml[
        [
            "station",
            "pollutant",
            "scenario",
            "selected_model_name",
            "Final_test_RMSE",
            "Final_test_MAE",
            "Final_test_R2",
        ]
    ].rename(
        columns={
            "selected_model_name": "Best ML model",
            "Final_test_RMSE": "Best ML RMSE",
            "Final_test_MAE": "Best ML MAE",
            "Final_test_R2": "Best ML R2",
        }
    )

    ridge = ridge[
        [
            "station",
            "pollutant",
            "scenario",
            "Final_test_RMSE",
            "Final_test_MAE",
            "Final_test_R2",
        ]
    ].rename(
        columns={
            "Final_test_RMSE": "Ridge RMSE",
            "Final_test_MAE": "Ridge MAE",
            "Final_test_R2": "Ridge R2",
        }
    )

    comparison = ml.merge(
        ridge,
        on=[
            "station",
            "pollutant",
            "scenario",
        ],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = comparison["_merge"] != "both"

    if unmatched.any():
        raise ValueError(
            "The Best-ML and Ridge files do not contain the "
            "same scenarios for "
            f"{station}-{pollutant}:\n"
            f"{comparison.loc[unmatched, ['scenario', '_merge']]}"
        )

    comparison = comparison.drop(
        columns="_merge"
    )

    return comparison



def build_pollutant_table(
    pollutant: str,
) -> pd.DataFrame:

    case_frames = []

    for station in STATIONS:

        if not case_is_available(
            station=station,
            pollutant=pollutant,
        ):
            print(
                f"Skipping unavailable case: "
                f"{station}-{pollutant}"
            )
            continue

        print(
            f"Reading: {station}-{pollutant}",
            flush=True,
        )

        case_table = load_case_metrics(
            station=station,
            pollutant=pollutant,
        )

        case_frames.append(case_table)

    if not case_frames:
        raise ValueError(
            f"No results found for pollutant {pollutant}."
        )

    table = pd.concat(
        case_frames,
        ignore_index=True,
    )

    expected_scenarios = set(SCENARIO_ORDER)

    for station, station_df in table.groupby("station"):

        found_scenarios = set(
            station_df["scenario"]
        )

        missing_scenarios = (
            expected_scenarios - found_scenarios
        )

        unexpected_scenarios = (
            found_scenarios - expected_scenarios
        )

        if missing_scenarios:
            print(
                f"WARNING: {station}-{pollutant} is missing "
                f"scenarios: {sorted(missing_scenarios)}"
            )

        if unexpected_scenarios:
            print(
                f"WARNING: {station}-{pollutant} contains "
                f"unexpected scenarios: "
                f"{sorted(unexpected_scenarios)}"
            )

    table["station_order"] = pd.Categorical(
        table["station"],
        categories=STATIONS,
        ordered=True,
    )

    table["scenario_order"] = pd.Categorical(
        table["scenario"],
        categories=SCENARIO_ORDER,
        ordered=True,
    )

    table = (
        table
        .sort_values(
            [
                "station_order",
                "scenario_order",
            ]
        )
        .reset_index(drop=True)
    )

    table["Station"] = (
        table["station"]
        .map(STATION_LABELS)
        .fillna(table["station"])
    )

    table["Scenario"] = (
        table["scenario"]
        .map(SCENARIO_LABELS)
        .fillna(table["scenario"])
    )

    table["Best ML model"] = (
        table["Best ML model"]
        .astype(str)
        .str.lower()
        .map(MODEL_LABELS)
        .fillna(table["Best ML model"])
    )

    table["Best ML RMSE"] = table[
        "Best ML RMSE"
    ].round(DECIMALS["RMSE"])

    table["Best ML MAE"] = table[
        "Best ML MAE"
    ].round(DECIMALS["MAE"])

    table["Best ML R2"] = table[
        "Best ML R2"
    ].round(DECIMALS["R2"])

    table["Ridge RMSE"] = table[
        "Ridge RMSE"
    ].round(DECIMALS["RMSE"])

    table["Ridge MAE"] = table[
        "Ridge MAE"
    ].round(DECIMALS["MAE"])

    table["Ridge R2"] = table[
        "Ridge R2"
    ].round(DECIMALS["R2"])

    final_columns = [
        "Station",
        "Scenario",
        "Best ML model",
        "Best ML RMSE",
        "Best ML MAE",
        "Best ML R2",
        "Ridge RMSE",
        "Ridge MAE",
        "Ridge R2",
    ]

    return table[final_columns]



def save_pollutant_table(
    pollutant: str,
    table: pd.DataFrame,
) -> None:

    complete_path = (
        OUTPUT_DIR
        / f"ML_vs_Ridge_{pollutant}_complete.csv"
    )

    table.to_csv(
        complete_path,
        index=False,
    )

    presentation = table.copy()

    presentation["Station"] = presentation[
        "Station"
    ].mask(
        presentation["Station"].duplicated(),
        "",
    )

    presentation_path = (
        OUTPUT_DIR
        / f"ML_vs_Ridge_{pollutant}_presentation.csv"
    )


    print(
        f"Saved complete table: {complete_path}",
        flush=True,
    )

    print(
        f"Saved presentation table: {presentation_path}",
        flush=True,
    )


def main() -> None:

    failed_pollutants = []

    for pollutant in POLLUTANTS:

        try:
            table = build_pollutant_table(
                pollutant=pollutant,
            )

            save_pollutant_table(
                pollutant=pollutant,
                table=table,
            )

            print(
                f"\nPreview for {pollutant}:"
            )
            print(
                table.to_string(index=False)
            )

        except Exception as error:

            failed_pollutants.append({
                "pollutant": pollutant,
                "error_type": type(error).__name__,
                "error_message": str(error),
            })

            print(
                f"ERROR processing {pollutant}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

    if failed_pollutants:

        failed_path = (
            OUTPUT_DIR
            / "failed_pollutant_tables.csv"
        )

        pd.DataFrame(
            failed_pollutants
        ).to_csv(
            failed_path,
            index=False,
        )

        print(
            f"\nSome tables failed. Details saved to: "
            f"{failed_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()