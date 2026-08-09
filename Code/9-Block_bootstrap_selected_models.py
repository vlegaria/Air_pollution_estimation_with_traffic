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
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.multitest import multipletests

RESULTS_ROOT = Path(r"Datasets_to_train")
OUTPUT_DIR = Path("block_bootstrap_results")
BLOCK_HOURS = 24
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
RANDOM_STATE = 754
ALPHA = 0.05

SCENARIO_COMPARISONS: List[Tuple[str, str]] = [
    ("baseline_tm", "traffic_tm"),
    ("nearest_bg_no_traffic", "nearest_bg_plus_traffic"),
    ("multi_station_no_traffic", "multi_station_plus_traffic"),
    ("traffic_tm", "nearest_bg_plus_traffic"),
    ("nearest_bg_plus_traffic", "multi_station_plus_traffic"),
    ("traffic_tm", "multi_station_plus_traffic"),
]

REQUIRED_PREDICTION_COLUMNS = [
    "station", "pollutant", "scenario", "model_name", "outer_fold",
    "row_id_original", "date", "y_true", "y_pred",
]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_name(value: object) -> str:
    text = str(value)
    for character in (" ", "/", "\\", "-", ":", "|"):
        text = text.replace(character, "_")
    return text


def comparison_seed(base_seed: int, station: str, pollutant: str,
                    scenario_a: str, scenario_b: str) -> int:
    key = f"{station}|{pollutant}|{scenario_a}|{scenario_b}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int((base_seed + int(digest[:8], 16)) % (2**32 - 1))


def validate_prediction_frame(predictions: pd.DataFrame, context: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in predictions.columns]
    if missing:
        raise ValueError(f"{context}: missing columns {missing}")

    result = predictions.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    for column in ["outer_fold", "row_id_original", "y_true", "y_pred"]:
        result[column] = pd.to_numeric(result[column], errors="raise")

    if result[REQUIRED_PREDICTION_COLUMNS].isna().any().any():
        counts = result[REQUIRED_PREDICTION_COLUMNS].isna().sum()
        raise ValueError(f"{context}: missing values:\n{counts[counts > 0]}")

    duplicate_key = ["station", "pollutant", "scenario", "outer_fold", "row_id_original"]
    duplicated = result.duplicated(subset=duplicate_key, keep=False)
    if duplicated.any():
        raise ValueError(f"{context}: duplicated prediction rows detected")

    return result.sort_values(
        ["station", "pollutant", "scenario", "outer_fold", "date"]
    ).reset_index(drop=True)


def load_final_test_predictions(results_root: Path) -> pd.DataFrame:
    files = sorted(results_root.rglob("final_test_predictions_*.csv"))
    files = [p for p in files if not p.name.startswith("ridge_final_test_predictions_")]
    if not files:
        raise FileNotFoundError(f"No final-test prediction files under {results_root.resolve()}")

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["_source_file"] = str(path)
        frames.append(frame)

    predictions = validate_prediction_frame(
        pd.concat(frames, ignore_index=True),
        "Combined final selected-model predictions",
    )

    if "split_role" in predictions.columns:
        unexpected = predictions.loc[
            predictions["split_role"] != "final_test_selected_model", "split_role"
        ].dropna().unique()
        if len(unexpected) > 0:
            warnings.warn(f"Unexpected split_role values: {unexpected.tolist()}")

    return predictions


def pair_scenario_predictions(case_predictions: pd.DataFrame,
                              scenario_a: str,
                              scenario_b: str) -> Tuple[pd.DataFrame, str, str]:
    key_columns = ["station", "pollutant", "outer_fold", "row_id_original", "date"]
    columns = key_columns + ["model_name", "y_true", "y_pred"]

    a = case_predictions.loc[case_predictions["scenario"] == scenario_a, columns].copy()
    b = case_predictions.loc[case_predictions["scenario"] == scenario_b, columns].copy()
    if a.empty:
        raise ValueError(f"Scenario unavailable: {scenario_a}")
    if b.empty:
        raise ValueError(f"Scenario unavailable: {scenario_b}")
    if a["model_name"].nunique() != 1 or b["model_name"].nunique() != 1:
        raise ValueError("Each scenario must have exactly one selected model")

    model_a = str(a["model_name"].iloc[0])
    model_b = str(b["model_name"].iloc[0])
    a = a.rename(columns={"y_true": "y_true_a", "y_pred": "y_pred_a"}).drop(columns="model_name")
    b = b.rename(columns={"y_true": "y_true_b", "y_pred": "y_pred_b"}).drop(columns="model_name")

    paired = a.merge(b, on=key_columns, how="inner", validate="one_to_one")
    if len(paired) != len(a) or len(paired) != len(b):
        raise ValueError(
            f"Samples differ for {scenario_b} vs {scenario_a}: "
            f"n_a={len(a)}, n_b={len(b)}, paired={len(paired)}"
        )
    if not np.allclose(paired["y_true_a"], paired["y_true_b"], rtol=0, atol=1e-12):
        raise ValueError("Observed targets differ between scenarios")

    paired = paired.rename(columns={"y_true_a": "y_true"}).drop(columns="y_true_b")
    return paired.sort_values(["outer_fold", "date"]).reset_index(drop=True), model_a, model_b


def build_non_overlapping_calendar_blocks(
    paired: pd.DataFrame, block_hours: int
) -> Dict[int, List[np.ndarray]]:
    if block_hours <= 0:
        raise ValueError("block_hours must be positive")

    blocks_by_fold: Dict[int, List[np.ndarray]] = {}
    for outer_fold, fold_data in paired.groupby("outer_fold", sort=True):
        fold_data = fold_data.sort_values("date").copy()
        if fold_data["date"].duplicated().any():
            raise ValueError(f"Duplicated dates in outer fold {outer_fold}")

        fold_start = fold_data["date"].min().floor("h")
        elapsed_hours = (fold_data["date"] - fold_start).dt.total_seconds() / 3600.0
        fold_data["_block_id"] = np.floor(elapsed_hours / block_hours).astype(int)
        blocks = [block.index.to_numpy(dtype=int)
                  for _, block in fold_data.groupby("_block_id", sort=True)]
        if len(blocks) < 2:
            raise ValueError(f"Only {len(blocks)} block(s) in outer fold {outer_fold}")
        blocks_by_fold[int(outer_fold)] = blocks

    return blocks_by_fold


def block_diagnostics(paired: pd.DataFrame,
                      blocks_by_fold: Dict[int, List[np.ndarray]]) -> Dict[str, float]:
    sizes = np.array([len(idx) for blocks in blocks_by_fold.values() for idx in blocks])
    return {
        "n_observations": int(len(paired)),
        "n_outer_folds": int(len(blocks_by_fold)),
        "n_blocks": int(len(sizes)),
        "block_size_min_observations": int(sizes.min()),
        "block_size_median_observations": float(np.median(sizes)),
        "block_size_max_observations": int(sizes.max()),
    }


def draw_block_bootstrap_indices(blocks_by_fold: Dict[int, List[np.ndarray]],
                                 rng: np.random.Generator) -> np.ndarray:
    sampled_parts: List[np.ndarray] = []
    for blocks in blocks_by_fold.values():
        positions = rng.integers(0, len(blocks), size=len(blocks))
        sampled_parts.extend(blocks[pos] for pos in positions)
    return np.concatenate(sampled_parts)


def bootstrap_two_scenarios(
    paired: pd.DataFrame,
    scenario_a: str,
    scenario_b: str,
    model_a: str,
    model_b: str,
    block_hours: int = BLOCK_HOURS,
    n_bootstrap: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Paired non-overlapping 24-hour block bootstrap.

    Effects are B - A. Negative values favour scenario B.
    The p-value uses a two-sided null-centred bootstrap distribution.
    """
    if n_bootstrap < 1000:
        warnings.warn("Fewer than 1,000 bootstrap replicates may be unstable")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be between 0 and 1")

    paired = paired.copy()
    paired["date"] = pd.to_datetime(paired["date"], errors="raise")
    required = ["outer_fold", "date", "y_true", "y_pred_a", "y_pred_b"]
    if paired[required].isna().any().any():
        raise ValueError("Paired predictions contain missing values")

    blocks_by_fold = build_non_overlapping_calendar_blocks(paired, block_hours)
    diagnostics = block_diagnostics(paired, blocks_by_fold)

    y_true = paired["y_true"].to_numpy(float)
    pred_a = paired["y_pred_a"].to_numpy(float)
    pred_b = paired["y_pred_b"].to_numpy(float)

    observed = {
        "MAE": {"a": mean_absolute_error(y_true, pred_a),
                "b": mean_absolute_error(y_true, pred_b)},
        "RMSE": {"a": rmse(y_true, pred_a), "b": rmse(y_true, pred_b)},
    }

    seed = comparison_seed(
        random_state,
        str(paired["station"].iloc[0]),
        str(paired["pollutant"].iloc[0]),
        scenario_a,
        scenario_b,
    )
    rng = np.random.default_rng(seed)

    distributions = {
        "MAE_difference_b_minus_a": np.empty(n_bootstrap),
        "MAE_relative_difference_percent": np.empty(n_bootstrap),
        "RMSE_difference_b_minus_a": np.empty(n_bootstrap),
        "RMSE_relative_difference_percent": np.empty(n_bootstrap),
    }

    for i in range(n_bootstrap):
        idx = draw_block_bootstrap_indices(blocks_by_fold, rng)
        yb, pa, pb = y_true[idx], pred_a[idx], pred_b[idx]
        mae_a, mae_b = mean_absolute_error(yb, pa), mean_absolute_error(yb, pb)
        rmse_a, rmse_b = rmse(yb, pa), rmse(yb, pb)
        mae_diff, rmse_diff = mae_b - mae_a, rmse_b - rmse_a
        distributions["MAE_difference_b_minus_a"][i] = mae_diff
        distributions["RMSE_difference_b_minus_a"][i] = rmse_diff
        distributions["MAE_relative_difference_percent"][i] = (
            100 * mae_diff / mae_a if mae_a != 0 else np.nan
        )
        distributions["RMSE_relative_difference_percent"][i] = (
            100 * rmse_diff / rmse_a if rmse_a != 0 else np.nan
        )

    alpha = 1 - ci_level
    rows = []
    for metric in ("MAE", "RMSE"):
        value_a = float(observed[metric]["a"])
        value_b = float(observed[metric]["b"])
        observed_diff = value_b - value_a
        observed_relative = 100 * observed_diff / value_a if value_a != 0 else np.nan

        diff_dist = distributions[f"{metric}_difference_b_minus_a"]
        rel_dist = distributions[f"{metric}_relative_difference_percent"]
        ci_low, ci_high = np.quantile(diff_dist, [alpha / 2, 1 - alpha / 2])
        finite_rel = rel_dist[np.isfinite(rel_dist)]
        if len(finite_rel):
            rel_ci_low, rel_ci_high = np.quantile(finite_rel, [alpha / 2, 1 - alpha / 2])
        else:
            rel_ci_low = rel_ci_high = np.nan

        centred = diff_dist - observed_diff
        p_value = (np.sum(np.abs(centred) >= abs(observed_diff)) + 1) / (len(centred) + 1)
        probability_b_better = (
            np.sum(diff_dist < 0) + 0.5 * np.sum(diff_dist == 0)
        ) / len(diff_dist)

        rows.append({
            "station": str(paired["station"].iloc[0]),
            "pollutant": str(paired["pollutant"].iloc[0]),
            "outer_fold": int(paired["outer_fold"].iloc[0])
                if paired["outer_fold"].nunique() == 1 else "multiple",
            "scenario_a": scenario_a,
            "scenario_b": scenario_b,
            "comparison": f"{scenario_b} vs {scenario_a}",
            "selected_model_a": model_a,
            "selected_model_b": model_b,
            "metric": metric,
            "value_a": value_a,
            "value_b": value_b,
            "difference_b_minus_a": float(observed_diff),
            "difference_ci_low": float(ci_low),
            "difference_ci_high": float(ci_high),
            "relative_difference_percent": float(observed_relative),
            "relative_difference_ci_low": float(rel_ci_low),
            "relative_difference_ci_high": float(rel_ci_high),
            "probability_b_better": float(probability_b_better),
            "pvalue_raw": float(p_value),
            "ci_level": float(ci_level),
            "block_hours": int(block_hours),
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_random_seed": int(seed),
            **diagnostics,
        })

    return pd.DataFrame(rows), distributions


def save_bootstrap_distributions(distributions: Dict[str, np.ndarray],
                                 summary: pd.DataFrame,
                                 output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    first = summary.iloc[0]
    filename = (
        f"{safe_name(first['station'])}__{safe_name(first['pollutant'])}__"
        f"{safe_name(first['scenario_b'])}_vs_{safe_name(first['scenario_a'])}.npz"
    )
    path = output_dir / filename
    metadata = {
        "station": first["station"],
        "pollutant": first["pollutant"],
        "scenario_a": first["scenario_a"],
        "scenario_b": first["scenario_b"],
        "selected_model_a": first["selected_model_a"],
        "selected_model_b": first["selected_model_b"],
        "block_hours": int(first["block_hours"]),
        "n_bootstrap": int(first["n_bootstrap"]),
        "ci_level": float(first["ci_level"]),
        "bootstrap_random_seed": int(first["bootstrap_random_seed"]),
        "difference_definition": "metric_b_minus_metric_a",
    }
    np.savez_compressed(path, **distributions, metadata_json=np.array(json.dumps(metadata)))
    return path


def add_holm_adjustments(results: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    result = results.copy()
    result["pvalue_holm_within_case"] = np.nan
    result["significant_holm_within_case"] = False

    for _, idx in result.groupby(["station", "pollutant", "metric"]).groups.items():
        idx = list(idx)
        reject, adjusted, _, _ = multipletests(
            result.loc[idx, "pvalue_raw"].to_numpy(float), alpha=alpha, method="holm"
        )
        result.loc[idx, "pvalue_holm_within_case"] = adjusted
        result.loc[idx, "significant_holm_within_case"] = reject

    result["pvalue_holm_global"] = np.nan
    result["significant_holm_global"] = False
    for _, idx in result.groupby("metric").groups.items():
        idx = list(idx)
        reject, adjusted, _, _ = multipletests(
            result.loc[idx, "pvalue_raw"].to_numpy(float), alpha=alpha, method="holm"
        )
        result.loc[idx, "pvalue_holm_global"] = adjusted
        result.loc[idx, "significant_holm_global"] = reject

    result["significant_ci"] = (
        (result["difference_ci_high"] < 0) | (result["difference_ci_low"] > 0)
    )
    return result


def run_block_bootstrap_analysis(
    predictions: pd.DataFrame,
    comparisons: Sequence[Tuple[str, str]],
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    distribution_dir = output_dir / "bootstrap_distributions"
    summaries: List[pd.DataFrame] = []
    skipped_records: List[dict] = []

    for (station, pollutant), case_predictions in predictions.groupby(
        ["station", "pollutant"], sort=True
    ):
        available = set(case_predictions["scenario"].unique())
        for scenario_a, scenario_b in comparisons:
            label = f"{scenario_b} vs {scenario_a}"
            missing = [s for s in (scenario_a, scenario_b) if s not in available]
            if missing:
                skipped_records.append({
                    "station": station,
                    "pollutant": pollutant,
                    "comparison": label,
                    "reason": "Missing scenario(s): " + ", ".join(missing),
                })
                continue

            try:
                paired, model_a, model_b = pair_scenario_predictions(
                    case_predictions, scenario_a, scenario_b
                )
                summary, distributions = bootstrap_two_scenarios(
                    paired, scenario_a, scenario_b, model_a, model_b
                )
                path = save_bootstrap_distributions(distributions, summary, distribution_dir)
                summary["bootstrap_distribution_file"] = str(path)
                summaries.append(summary)
                print(f"Completed: {station} | {pollutant} | {label}", flush=True)
            except Exception as error:
                skipped_records.append({
                    "station": station,
                    "pollutant": pollutant,
                    "comparison": label,
                    "reason": f"{type(error).__name__}: {error}",
                })
                warnings.warn(f"Skipped {station} | {pollutant} | {label}: {error}")

    if not summaries:
        raise RuntimeError("No block-bootstrap comparison was completed")

    results = add_holm_adjustments(pd.concat(summaries, ignore_index=True))
    results = results.sort_values(
        ["station", "pollutant", "metric", "scenario_a", "scenario_b"]
    ).reset_index(drop=True)
    return results, pd.DataFrame(skipped_records)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = load_final_test_predictions(RESULTS_ROOT)
    results, skipped = run_block_bootstrap_analysis(
        predictions, SCENARIO_COMPARISONS, OUTPUT_DIR
    )

    full_path = OUTPUT_DIR / "block_bootstrap_summary.csv"
    results.to_csv(full_path, index=False)

    compact_columns = [
        "station", "pollutant", "comparison", "scenario_a", "scenario_b",
        "selected_model_a", "selected_model_b", "metric", "value_a", "value_b",
        "difference_b_minus_a", "difference_ci_low", "difference_ci_high",
        "relative_difference_percent", "relative_difference_ci_low",
        "relative_difference_ci_high", "probability_b_better", "pvalue_raw",
        "pvalue_holm_within_case", "pvalue_holm_global", "significant_ci",
        "significant_holm_within_case", "significant_holm_global",
        "n_observations", "n_blocks", "block_hours", "n_bootstrap",
    ]

    
    scenarios_map = {
    "multi_station_plus_traffic vs multi_station_no_traffic": "scen3 vs scen3-noT",
    "nearest_bg_plus_traffic vs nearest_bg_no_traffic": "scen2 vs scen2-noT",
    "traffic_tm vs baseline_tm": "scen1 vs scen1-noT",
    "nearest_bg_plus_traffic vs traffic_tm": "scen2 vs scen1",
    "multi_station_plus_traffic vs nearest_bg_plus_traffic": "scen3 vs scen2",
    "multi_station_plus_traffic vs traffic_tm": "scen3 vs scen1"
    }

    results["comparison"] = results["comparison"].apply(lambda x: scenarios_map.get(x, x))

    results[compact_columns].to_csv(
        OUTPUT_DIR / "block_bootstrap_summary_compact.csv", index=False
    )
    if not skipped.empty:
        skipped.to_csv(OUTPUT_DIR / "skipped_comparisons.csv", index=False)

    config = {
        "results_root": str(RESULTS_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "block_hours": BLOCK_HOURS,
        "n_bootstrap": N_BOOTSTRAP,
        "ci_level": CI_LEVEL,
        "random_state": RANDOM_STATE,
        "alpha": ALPHA,
        "comparisons": [{"scenario_a": a, "scenario_b": b}
                        for a, b in SCENARIO_COMPARISONS],
        "difference_definition": "metric_b_minus_metric_a",
        "primary_adjustment": "Holm within station-pollutant-metric",
        "sensitivity_adjustment": "Holm globally, separately by metric",
        "bootstrap_type": "paired non-overlapping 24-hour chronological block bootstrap",
    }
    with open(OUTPUT_DIR / "block_bootstrap_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Full summary: {full_path.resolve()}")
    print(f"Distributions: {(OUTPUT_DIR / 'bootstrap_distributions').resolve()}")


if __name__ == "__main__":
    main()
