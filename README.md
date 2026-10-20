# QoL and overall-survival benefit prediction

Reproducible code for a trial-level machine-learning analysis that evaluates
whether quality-of-life (QoL) endpoint data add predictive information for
overall-survival (OS) benefit beyond other non-OS endpoints in randomized
oncology trial comparisons.

## Final model

- L2-regularized logistic regression (`C=1`)
- Direction-harmonized endpoint effects, where positive values favor the active arm
- Two prespecified feature sets: other endpoints alone and other endpoints plus QoL
- Twenty repeats of five-fold stratified group cross-validation
- All comparison rows from one NCT kept in the same fold
- Equal total weight for each unique NCT
- Fold-local imputation, scaling, and categorical encoding
- Held-out linear SHAP values for model interpretation

The source database and the processed review workbook used for the final
analysis are available in [`data/raw`](data/raw) and
[`data/processed`](data/processed), respectively. The modeling table is the
`Model_Analysis_Data` sheet of the processed workbook; its required columns are
documented in [data/README.md](data/README.md).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the repeated grouped validation

```bash
python src/run_model.py \
  --data data/analysis_table_direction_adjusted.csv \
  --output outputs/model
```

The command exports held-out predictions, mean out-of-fold predictions, ROC
coordinates, point metrics, and NCT-cluster bootstrap confidence intervals.

## Run the held-out SHAP analysis

```bash
python src/run_shap.py \
  --data data/analysis_table_direction_adjusted.csv \
  --output outputs/shap
```

SHAP values are computed only for held-out observations. Preprocessing and the
SHAP background data are fitted separately within each training fold. SHAP
importance describes model influence and should not be interpreted as causal.

## Outcome-leakage safeguards

OS result fields and OS effect estimates must not be included as predictors.
`target_os_benefit` is used only as the binary outcome. NCT identifiers are used
only for grouped splitting and weighting. The scripts stop if required grouping,
outcome, or feature columns are missing.

## Reproducibility

The default random seed, cross-validation schedule, model specification, and
feature definitions match the final analysis. Command-line options can reduce
the number of repeats or bootstrap samples for a quick test, but manuscript
results should use the defaults.
