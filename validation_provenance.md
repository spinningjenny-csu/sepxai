# MIMIC-III validation-extract provenance

The R2 analysis imports `0 原始数据们/MIMIC.csv`, an analytic derivative of
the supplementary dataset published with Hou et al., *Journal of Translational
Medicine* 2020;18:462 (doi: `10.1186/s12967-020-02620-5`; PMID: `33287854`;
PMC: `PMC7720497`).

- Source supplement: `12967_2020_2620_MOESM1_ESM.csv`
- Source URL: `https://static-content.springer.com/esm/art%3A10.1186%2Fs12967-020-02620-5/MediaObjects/12967_2020_2620_MOESM1_ESM.csv`
- Source supplement dimensions: 4,559 rows and 106 columns
- Source supplement SHA-256: `6e0b4104ec2388fcef25f179c92c449b567a6ef7a11b17276b77e05a0c6f32de`
- Supplied analytic-file dimensions: 4,559 rows and 91 columns
- Supplied analytic-file SHA-256: `f1a684b761760cf78c100b4d7238acb4be7cbbb15d19134ac36736d9c7cafea4`

The retained values match the corresponding source-supplement fields
row-for-row. Direct identifiers, timestamps, database-source fields, and
duplicated identifier columns were removed before the file was supplied; the
source `urineoutput` field is named `_` in the analytic derivative and is not a
SepXAI predictor.

For a public-code rerun, the 106-column source supplement itself may be saved as
`data/MIMIC.csv`. The loader selects fields by name, ignores unrelated columns,
and recognizes the verified SHA-256 of either the source supplement or the
reported 91-column analytic derivative.

The exact reproducible steps performed after import are implemented in
`SepXAI.py`:

- outcome: `thirtyday_expire_flag`;
- age: `age`;
- sex: `is_male`;
- race: `race_white`, `race_black`, and `race_hispanic`;
- severity score: `sofa`;
- predictor mapping: `lactate_min` → minimum lactate, `sysbp_min` → minimum
  non-invasive systolic blood pressure, `resprate_min` → minimum respiratory
  rate, `diasbp_max` → maximum non-invasive diastolic blood pressure,
  `heartrate_mean` → mean heart rate, `spo2_min` → minimum SpO₂, and
  `tempc_max` → maximum temperature;
- unit and physiological-range checks identical to the MIMIC-IV model schema;
- +24-hour landmark: retain records with `icu_los >= 1.0` day before creating
  the external outcome and predictor arrays;
- median imputation by the locked MIMIC-IV pipeline;
- evaluation only, with no feature selection, fitting, tuning, calibration, or
  threshold selection.

The complete column mapping, range checks, and landmark logic are in
`MIMIC_III_RENAME`, `VALID_RANGES`, and `load_mimic_iii()` in `SepXAI.py`.
