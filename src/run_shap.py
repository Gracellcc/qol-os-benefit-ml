#!/usr/bin/env python3
"""Held-out linear SHAP analysis for the final Other-endpoints-plus-QoL model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedGroupKFold

from run_model import build_pipeline, engineered_table, feature_sets, validate_input


def base_feature(transformed: str, originals: list[str]) -> str:
    shown = transformed.removeprefix("missingindicator_")
    for original in sorted(originals, key=len, reverse=True):
        if shown == original or shown.startswith(original + "_"):
            return original
    return shown


def feature_domain(feature: str) -> str:
    if feature.startswith("qol_"):
        return "QoL"
    if feature.startswith("other_"):
        return "Other endpoint"
    return "Cross-domain"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20271001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = engineered_table(pd.read_csv(args.data))
    sets = feature_sets()
    validate_input(data, sets)
    features = sets["Other endpoints + QoL"]
    y = data.target_os_benefit.to_numpy(int)
    groups = data.CV_Group_ID.to_numpy()
    weights = data.nct_equal_weight.to_numpy(float)
    rows = []

    for repeat in range(args.repeats):
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed + repeat)
        for fold, (train, test) in enumerate(splitter.split(np.zeros((len(data), 1)), y, groups)):
            model = build_pipeline(data, features, args.seed + repeat * 100 + fold)
            model.fit(data.loc[train, features], y[train], model__sample_weight=weights[train])
            transform = model.named_steps["preprocess"]
            estimator = model.named_steps["model"]
            x_train = np.asarray(transform.transform(data.loc[train, features]), dtype=float)
            x_test = np.asarray(transform.transform(data.loc[test, features]), dtype=float)
            names = transform.get_feature_names_out().tolist()
            values = np.asarray(shap.LinearExplainer(estimator, x_train).shap_values(x_test), dtype=float)
            for local_row, source_index in enumerate(test):
                rows.extend({
                    "repeat": repeat, "fold": fold, "row_index": int(source_index),
                    "RowID": data.iloc[source_index].RowID, "NCT_ID": data.iloc[source_index].NCT_ID,
                    "transformed_feature": name,
                    "base_feature": base_feature(name, features),
                    "domain": feature_domain(base_feature(name, features)),
                    "transformed_value": float(x_test[local_row, column]),
                    "shap_value_log_odds": float(values[local_row, column]),
                } for column, name in enumerate(names))

    long = pd.DataFrame(rows)
    grouped_per_row = long.groupby(
        ["repeat", "row_index", "RowID", "NCT_ID", "base_feature", "domain"], as_index=False
    ).shap_value_log_odds.sum()
    importance = grouped_per_row.groupby(["base_feature", "domain"], as_index=False).agg(
        mean_absolute_shap=("shap_value_log_odds", lambda values: float(np.mean(np.abs(values)))),
        mean_signed_shap=("shap_value_log_odds", "mean"),
    ).sort_values("mean_absolute_shap", ascending=False).reset_index(drop=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    importance["displayed_in_bar_plot"] = importance["rank"].le(20)
    importance["percent_of_total_mean_absolute_shap"] = (
        100 * importance.mean_absolute_shap / importance.mean_absolute_shap.sum()
    )

    long.to_csv(args.output / "heldout_shap_values_long.csv", index=False)
    importance.to_csv(args.output / "grouped_shap_importance.csv", index=False)

    top = importance.head(20).sort_values("mean_absolute_shap")
    colors = {"QoL": "#4472C4", "Other endpoint": "#7F7F7F", "Cross-domain": "#8064A2"}
    figure, axis = plt.subplots(figsize=(9, 7.2))
    axis.barh(top.base_feature, top.mean_absolute_shap,
              color=[colors[domain] for domain in top.domain], edgecolor="white", linewidth=.5)
    axis.set_xlabel("Mean absolute held-out SHAP value (log-odds)")
    axis.set_title("Direction-adjusted ridge model: grouped SHAP importance")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=.18)
    figure.tight_layout()
    figure.savefig(args.output / "shap_grouped_importance.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(importance.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
