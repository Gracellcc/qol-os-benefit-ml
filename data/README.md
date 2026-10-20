# Study data and input schema

The repository contains:

- `raw/Combined_2013-2021_8.21.xlsx`: source database received for cleaning
- `processed/Clean_Dataset_Review_Final.xlsx`: cleaning audit, exclusions,
  direction harmonization, and final analysis table

The `Model_Analysis_Data` sheet in the processed workbook contains one row per
analytic trial comparison. Export this sheet as
`data/analysis_table_direction_adjusted.csv` before running the scripts. The
modeling scripts expect the following identifiers:

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
