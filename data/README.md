# Input data schema

The analysis dataset is not distributed in this repository. The modeling scripts
expect one row per analytic trial comparison and the following identifiers:

- `RowID`: unique comparison identifier
- `NCT_ID`: normalized ClinicalTrials.gov identifier
- `CV_Group_ID`: grouping identifier; all rows with the same value remain together
- `target_os_benefit`: binary outcome (`1` for OS benefit, `0` otherwise)
- `nct_equal_weight`: row weight such that weights sum to one within each NCT

The required predictor columns are defined in `src/run_model.py` as
`OTHER_BASE_FEATURES`, `QOL_BASE_FEATURES`, and the fold-independent engineered
features created by `engineered_table()`.

Direction-harmonized numeric effects must follow this convention: a positive
value favors the active arm. Examples include higher or longer favorable outcomes,
`-log(HR)` for hazard ratios, and control-minus-active event incidence when lower
event incidence is favorable.

OS effect sizes, OS p-values, drug names, and NCT identifiers are not model
predictors.
