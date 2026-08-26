#!/usr/bin/env python3
"""Final direction-adjusted ridge model with repeated NCT-grouped validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OTHER_BASE_FEATURES = [
    "other_benefit_status", "other_endpoint_type", "other_time_unit",
    "other_stat_type", "other_symmetric_difference", "other_neg_log10_p",
    "other_log_total_n",
]
QOL_BASE_FEATURES = [
    "qol_benefit_status", "qol_measure_mode", "qol_unit",
    "qol_instrument_family", "qol_stat_type", "qol_symmetric_difference",
    "qol_neg_log10_p", "qol_log_total_n",
]
OTHER_ENGINEERED_FEATURES = [
    "other_effect_x_significance", "other_abs_effect_x_significance",
    "other_p_below_005", "other_n_log_ratio", "other_relative_uncertainty",
    "other_missing_count",
]
COMBINED_ENGINEERED_FEATURES = [
    "qol_effect_x_significance", "qol_abs_effect_x_significance",
    "qol_p_below_005", "qol_n_log_ratio", "qol_relative_uncertainty",
    "qol_missing_count", "joint_benefit_status", "effect_product",
    "evidence_product", "absolute_effect_sum", "log_sample_size_difference",
    "both_p_below_005",
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def engineered_table(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    for prefix in ["other", "qol"]:
        effect = frame[f"{prefix}_symmetric_difference"]
        strength = frame[f"{prefix}_neg_log10_p"].clip(upper=12)
        frame[f"{prefix}_effect_x_significance"] = effect * strength
        frame[f"{prefix}_abs_effect_x_significance"] = effect.abs() * strength
        frame[f"{prefix}_p_below_005"] = frame[f"{prefix}_p_value"].lt(.05).astype(float)
        frame.loc[frame[f"{prefix}_p_value"].isna(), f"{prefix}_p_below_005"] = np.nan
        frame[f"{prefix}_n_log_ratio"] = np.log(safe_divide(
            frame[f"{prefix}_active_n"] + 1, frame[f"{prefix}_control_n"] + 1
        ))
        uncertainty = frame[f"{prefix}_active_uncertainty"].abs() + frame[f"{prefix}_control_uncertainty"].abs()
        magnitude = frame[f"{prefix}_active_value"].abs() + frame[f"{prefix}_control_value"].abs() + 1e-8
        frame[f"{prefix}_relative_uncertainty"] = safe_divide(uncertainty, magnitude)
        source = [
            f"{prefix}_symmetric_difference", f"{prefix}_neg_log10_p",
            f"{prefix}_log_total_n", f"{prefix}_relative_uncertainty",
        ]
        frame[f"{prefix}_missing_count"] = frame[source].isna().sum(axis=1)
    frame["joint_benefit_status"] = (
        frame.other_benefit_status.astype(str) + " / " + frame.qol_benefit_status.astype(str)
    )
    frame["effect_product"] = frame.other_symmetric_difference * frame.qol_symmetric_difference
    frame["evidence_product"] = frame.other_effect_x_significance * frame.qol_effect_x_significance
    frame["absolute_effect_sum"] = frame.other_symmetric_difference.abs() + frame.qol_symmetric_difference.abs()
    frame["log_sample_size_difference"] = frame.other_log_total_n - frame.qol_log_total_n
    frame["both_p_below_005"] = frame.other_p_below_005 * frame.qol_p_below_005
    return frame


def feature_sets() -> dict[str, list[str]]:
    other = list(dict.fromkeys(OTHER_BASE_FEATURES + OTHER_ENGINEERED_FEATURES))
    combined = list(dict.fromkeys(
        OTHER_BASE_FEATURES + QOL_BASE_FEATURES + OTHER_ENGINEERED_FEATURES + COMBINED_ENGINEERED_FEATURES
    ))
    return {"Other endpoints only": other, "Other endpoints + QoL": combined}


def validate_input(data: pd.DataFrame, sets: dict[str, list[str]]) -> None:
    required = {"RowID", "NCT_ID", "CV_Group_ID", "target_os_benefit", "nct_equal_weight"}
    required.update(feature for features in sets.values() for feature in features)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not data.target_os_benefit.isin([0, 1]).all():
        raise ValueError("target_os_benefit must contain only 0 and 1")
    if data.RowID.duplicated().any():
        raise ValueError("RowID must be unique")
    weight_sum = data.groupby("CV_Group_ID").nct_equal_weight.sum()
    if not np.allclose(weight_sum, 1):
        raise ValueError("nct_equal_weight must sum to one within each CV_Group_ID")
    forbidden = [feature for features in sets.values() for feature in features
                 if feature == "target_os_benefit" or feature.casefold().startswith("os_")]
    if forbidden:
        raise ValueError(f"Outcome-leakage features detected: {sorted(set(forbidden))}")


def preprocessor(frame: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    categorical = [column for column in features if frame[column].dtype == "object"]
    numeric = [column for column in features if column not in categorical]
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=4, sparse_output=False)),
        ]), categorical),
    ], verbose_feature_names_out=False)


def build_pipeline(frame: pd.DataFrame, features: list[str], seed: int) -> Pipeline:
    return Pipeline([
        ("preprocess", preprocessor(frame, features)),
        ("model", LogisticRegression(C=1, l1_ratio=0, solver="saga", max_iter=6000, random_state=seed)),
    ])


def metric_values(y: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    prediction = probability >= .5
    negative = y == 0
    true_negative = float(np.sum(weight[negative] * (~prediction[negative])))
    specificity = true_negative / float(np.sum(weight[negative]))
    predicted_negative = float(np.sum(weight[~prediction]))
    npv = true_negative / predicted_negative if predicted_negative else np.nan
    return {
        "roc_auc": roc_auc_score(y, probability, sample_weight=weight),
        "pr_auc": average_precision_score(y, probability, sample_weight=weight),
        "brier": brier_score_loss(y, probability, sample_weight=weight),
        "log_loss": log_loss(y, np.clip(probability, 1e-8, 1 - 1e-8), sample_weight=weight),
        "accuracy": accuracy_score(y, prediction, sample_weight=weight),
        "precision": precision_score(y, prediction, sample_weight=weight, zero_division=0),
        "npv": npv,
        "sensitivity": recall_score(y, prediction, sample_weight=weight, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y, prediction, sample_weight=weight, zero_division=0),
    }


def repeated_group_predictions(data: pd.DataFrame, sets: dict[str, list[str]], repeats: int,
                               folds: int, seed: int) -> pd.DataFrame:
    y = data.target_os_benefit.to_numpy(int)
    groups = data.CV_Group_ID.to_numpy()
    weights = data.nct_equal_weight.to_numpy(float)
    rows = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        for fold, (train, test) in enumerate(splitter.split(np.zeros((len(data), 1)), y, groups)):
            for name, features in sets.items():
                model = build_pipeline(data, features, seed + repeat * 100 + fold)
                model.fit(data.loc[train, features], y[train], model__sample_weight=weights[train])
                probabilities = model.predict_proba(data.loc[test, features])[:, 1]
                rows.extend({
                    "feature_set": name, "repeat": repeat, "fold": fold,
                    "row_index": int(index), "RowID": data.iloc[index].RowID,
                    "NCT_ID": data.iloc[index].NCT_ID, "target": int(y[index]),
                    "weight": float(weights[index]), "probability": float(probabilities[position]),
                } for position, index in enumerate(test))
    return pd.DataFrame(rows)


def mean_oof(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.groupby(
        ["feature_set", "row_index", "RowID", "NCT_ID", "target", "weight"], as_index=False
    ).agg(probability=("probability", "mean"), prediction_sd=("probability", "std"),
          n_predictions=("probability", "size"))


def roc_coordinates(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame in oof.groupby("feature_set", sort=False):
        fpr, tpr, thresholds = roc_curve(frame.target, frame.probability,
                                         sample_weight=frame.weight, drop_intermediate=False)
        rows.extend({
            "feature_set": name, "curve_point": index + 1,
            "false_positive_rate": float(x), "true_positive_rate": float(y),
            "specificity": float(1 - x),
            "threshold": np.nan if np.isinf(threshold) else float(threshold),
        } for index, (x, y, threshold) in enumerate(zip(fpr, tpr, thresholds)))
    return pd.DataFrame(rows)


def bootstrap_metrics(oof: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = np.sort(oof.NCT_ID.unique())
    group_index = {group: index for index, group in enumerate(groups)}
    frames = {name: frame.copy() for name, frame in oof.groupby("feature_set", sort=False)}
    indices = {name: frame.NCT_ID.map(group_index).to_numpy() for name, frame in frames.items()}
    point = {name: metric_values(frame.target.to_numpy(), frame.probability.to_numpy(), frame.weight.to_numpy())
             for name, frame in frames.items()}
    records = []
    for iteration in range(iterations):
        counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        for name, frame in frames.items():
            values = metric_values(frame.target.to_numpy(), frame.probability.to_numpy(),
                                   frame.weight.to_numpy() * counts[indices[name]])
            records.extend({"iteration": iteration, "feature_set": name, "metric": metric, "value": value}
                           for metric, value in values.items())
    bootstrap = pd.DataFrame(records)
    summary = []
    for (name, metric), frame in bootstrap.groupby(["feature_set", "metric"], sort=False):
        summary.append({
            "feature_set": name, "metric": metric, "estimate": point[name][metric],
            "ci_lower_95pct": frame.value.quantile(.025),
            "ci_upper_95pct": frame.value.quantile(.975),
            "classification_threshold": .5 if metric in {
                "accuracy", "precision", "npv", "sensitivity", "specificity", "f1"
            } else np.nan,
        })
    return pd.DataFrame(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20271001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = engineered_table(pd.read_csv(args.data))
    sets = feature_sets()
    validate_input(data, sets)
    predictions = repeated_group_predictions(data, sets, args.repeats, args.folds, args.seed)
    oof = mean_oof(predictions)
    metrics = bootstrap_metrics(oof, args.bootstrap, args.seed + 30000)
    predictions.to_csv(args.output / "heldout_predictions.csv", index=False)
    oof.to_csv(args.output / "mean_oof_predictions.csv", index=False)
    roc_coordinates(oof).to_csv(args.output / "roc_curve_data.csv", index=False)
    metrics.to_csv(args.output / "model_metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
