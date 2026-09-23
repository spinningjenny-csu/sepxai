# SepXAI

Reproducibility code accompanying the study:

**SepXAI: Performance, Calibration, and Safety Boundary Evaluation of Explainable AI for ICU Sepsis Mortality Prediction**

SepXAI is a research model for estimating 30-day mortality risk among ICU patients with sepsis. The model was developed using MIMIC-IV and evaluated through a same-center cross-database assessment using MIMIC-III.

## Intended use

This repository is provided for research and reproducibility purposes only.

SepXAI is not a medical device and has not been validated for treatment selection or individual clinical decision-making. SHAP and LIME outputs describe model associations and local input sensitivity; they do not estimate causal treatment effects.

## Prediction setting

- Development index time: first ICU admission during the first hospitalization
- Development observation window: ICU admission -6 h to ICU admission +24 h
- Prediction time: ICU admission +24 h
- Outcome: 30-day all-cause mortality from hospital admission
- Random seed: 42
- Operating threshold: 0.12454709065853185
- Threshold rule: highest-specificity threshold achieving sensitivity >=0.80 in the held-out calibration subset

## Final model inputs

The final locked model uses seven predictors:

1. Minimum lactate, mmol/L
2. Minimum non-invasive systolic blood pressure, mmHg
3. Minimum respiratory rate, breaths/min
4. Maximum non-invasive diastolic blood pressure, mmHg
5. Mean heart rate, beats/min
6. Minimum peripheral oxygen saturation, %
7. Maximum temperature, degrees C

## Repository contents

- `SepXAI.py`: preprocessing, feature selection, model development, calibration, evaluation, safety-boundary analysis, SHAP/LIME analysis, figure generation, and the Streamlit research interface
- `SepXAI.sql`: time-filtered MIMIC-IV cohort and predictor extraction query
- `requirements_R2.txt`: locked Python package versions
- `MIMIC_III_validation_provenance.md`: validation-source provenance, file checksum, predictor mapping, and +24-hour landmark rule
- `model_metadata_public.json`: non-sensitive metadata identifying the reported model and code release
- `.gitignore`: rules that prevent restricted data and generated model artifacts from being committed

## Data access

Direct access to the MIMIC databases requires PhysioNet credentials. The MIMIC-III external-validation analytic source used here is also available as the public Hou et al. supplementary CSV linked in `MIMIC_III_validation_provenance.md`. Patient-level MIMIC-IV extracts, patient-level predictions, and the trained model artifact are not distributed through this public repository.

The code expects the following local input structure:

```text
data/
|-- MIMIC IV/
|   |-- MIMIC_01.csv
|   `-- MIMIC_02.csv
`-- MIMIC.csv
```

Users are responsible for obtaining the necessary PhysioNet credentials and complying with the applicable data use agreement.

### MIMIC-III validation provenance and landmark

The MIMIC-III validation file is the analytic supplement to Hou et al., *Journal of Translational Medicine* 2020;18:462 (doi: `10.1186/s12967-020-02620-5`), source supplement `12967_2020_2620_MOESM1_ESM.csv`. The supplied local analytic file preserves the published clinical variables used here and has SHA-256 `f1a684b761760cf78c100b4d7238acb4be7cbbb15d19134ac36736d9c7cafea4`.

The published vital-sign and laboratory fields are first-day ICU summaries. The public source supplement can be saved directly as `data/MIMIC.csv`; the loader uses named fields, ignores unrelated columns, and recognizes the verified checksum of either the 106-column source supplement or the reported 91-column analytic derivative. `load_mimic_iii()` applies the +24-hour ICU landmark directly as `icu_los >= 1.0` day before outcome and predictor arrays are created. The previous hospital-length-of-stay proxy is not used. MIMIC-III remains evaluation only; it does not contribute to feature selection, model fitting, calibration, or threshold choice.

## Installation

Python 3.12.4 was used for the reported R2 analysis.

```bash
git clone https://github.com/andrelau0622/30_days_prediction.git
cd 30_days_prediction

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_R2.txt
```

## Reproduce the analysis

After placing the credentialed input files in a local data directory:

```bash
python SepXAI.py \
  --data-root "/path/to/credentialed/data" \
  --output-root "./analysis_results"
```

For a shorter software check:

```bash
python SepXAI.py \
  --data-root "/path/to/credentialed/data" \
  --output-root "./analysis_results" \
  --quick
```

The complete run generates tables, figures, the locally trained model artifact, and machine-readable metadata under `analysis_results/`.

## Run the research interface

The full analysis must first be completed by an authorized user so that the model and metadata files exist locally.

```bash
streamlit run SepXAI.py
```

If the analysis output is stored elsewhere:

```bash
export SEPXAI_RESULTS_DIR="/path/to/analysis_results"
streamlit run SepXAI.py
```

Changing an entered value represents a what-if input perturbation of the model. It does not simulate treatment response or estimate the clinical effect of an intervention.

## Model identification and controlled access

The exact reported model is identified in `model_metadata_public.json` by its code release and artifact checksum. The trained MIMIC-derived model itself is not publicly distributed through GitHub.

Any sharing of the trained artifact or patient-level derived data must follow the applicable PhysioNet credentialed-data requirements.

## Citation

Please cite the associated manuscript when using this code.
