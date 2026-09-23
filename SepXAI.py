#!/usr/bin/env python3
"""Complete leakage-audited SepXAI analysis and research interface.

This script rebuilds the manuscript analysis directly from the three supplied
CSV extracts. It deliberately does not reuse the previously stacked MICE data
or any cached notebook model.

Primary design implemented for R2
---------------------------------
- Cohort: one first hospital/first ICU record per patient, as encoded by the
  supplied MIMIC-IV extraction.
- Development observation window: ICU admission -6 h through ICU admission
  +24 h.
- Prediction time: end of the 24-h observation window.
- Landmark population: patients alive at the prediction time.
- Outcome: 30-day all-cause mortality from hospital admission, matching the
  available derivation and validation outcome fields.
- Development/internal test split: patient level, 80/20, random seed 42.
- Development data are further divided into model-training and
  calibration/threshold subsets.
- Median imputation and any standardization are fit inside training pipelines.
- The held-out MIMIC-III extract contains published first-day ICU summary
  variables and is never used for model fitting, tuning, calibration, feature
  selection, or threshold selection.

Direct Python execution writes tables, figures, a locked model, and
machine-readable metadata. Streamlit execution launches the research interface
from the same source file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import shap
import sklearn
import statsmodels
import statsmodels.api as sm
import xgboost
import catboost
import lightgbm
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message="Inconsistent values: penalty=l1",
    category=UserWarning,
)

SEED = 42
N_BOOTSTRAP = 1000
TARGET_SENSITIVITY = 0.80
MODEL_VERSION = "SepXAI-R2-2026-09-02"
MIMIC_III_SOURCE_DOI = "10.1186/s12967-020-02620-5"
MIMIC_III_SOURCE_FILE = "12967_2020_2620_MOESM1_ESM.csv"
MIMIC_III_SOURCE_SHA256 = (
    "6e0b4104ec2388fcef25f179c92c449b567a6ef7a11b17276b77e05a0c6f32de"
)
MIMIC_III_ANALYTIC_SHA256 = (
    "f1a684b761760cf78c100b4d7238acb4be7cbbb15d19134ac36736d9c7cafea4"
)

FEATURES = [
    "lactate_min",
    "sbp_ni_min",
    "spo2_max",
    "resp_rate_min",
    "dbp_ni_max",
    "heart_rate_mean",
    "spo2_min",
    "temperature_max",
]

DISPLAY_NAMES = {
    "urine_output": "Urine output",
    "lactate_min": "Minimum lactate",
    "sbp_ni_min": "Minimum non-invasive SBP",
    "spo2_max": "Maximum SpO2",
    "resp_rate_min": "Minimum respiratory rate",
    "dbp_ni_max": "Maximum non-invasive DBP",
    "heart_rate_mean": "Mean heart rate",
    "spo2_min": "Minimum SpO2",
    "temperature_max": "Maximum temperature",
}

UNITS = {
    "urine_output": "mL/24 h",
    "lactate_min": "mmol/L",
    "sbp_ni_min": "mmHg",
    "spo2_max": "%",
    "resp_rate_min": "breaths/min",
    "dbp_ni_max": "mmHg",
    "heart_rate_mean": "beats/min",
    "spo2_min": "%",
    "temperature_max": "degrees C",
}

# Conservative plausibility limits applied before imputation and aggregation.
VALID_RANGES = {
    "lactate_min": (0.2, 30.0),
    "sbp_ni_min": (30.0, 300.0),
    "spo2_max": (30.0, 100.0),
    "resp_rate_min": (1.0, 80.0),
    "dbp_ni_max": (10.0, 200.0),
    "heart_rate_mean": (20.0, 250.0),
    "spo2_min": (30.0, 100.0),
    "temperature_max": (25.0, 45.0),
}

FIGURE_PALETTE = {
    "cream": "#EFE8DD",
    "navy": "#345282",
    "gold": "#DCAF75",
    "periwinkle": "#9CABCC",
    "green": "#48614F",
}

MODEL_STYLES = {
    "Logistic regression": {
        "color": FIGURE_PALETTE["navy"],
        "linestyle": "-",
        "marker": "o",
    },
    "SVM": {
        "color": FIGURE_PALETTE["navy"],
        "linestyle": "--",
        "marker": "s",
    },
    "XGBoost": {
        "color": FIGURE_PALETTE["gold"],
        "linestyle": "-",
        "marker": "^",
    },
    "LightGBM": {
        "color": FIGURE_PALETTE["gold"],
        "linestyle": "--",
        "marker": "D",
    },
    "AdaBoost": {
        "color": FIGURE_PALETTE["periwinkle"],
        "linestyle": ":",
        "marker": "P",
    },
    "CatBoost": {
        "color": FIGURE_PALETTE["green"],
        "linestyle": "-.",
        "marker": "X",
    },
}

SHORT_NAMES = {
    "lactate_min": "Lactate, min",
    "sbp_ni_min": "SBP, min",
    "spo2_max": "SpO2, max",
    "resp_rate_min": "Respiratory rate, min",
    "dbp_ni_max": "DBP, max",
    "heart_rate_mean": "Heart rate, mean",
    "spo2_min": "SpO2, min",
    "temperature_max": "Temperature, max",
}

MIMIC_IV_RENAME = {
    "尿量(单位ml）": "urine_output",
    "lactate血气乳酸小": "lactate_min",
    "sbp_ni无创收缩压小": "sbp_ni_min",
    "spo2氧饱和度大": "spo2_max",
    "resp_rate呼吸小": "resp_rate_min",
    "dbp_ni无创舒张压大": "dbp_ni_max",
    "vitalhr心率\n": "heart_rate_mean",
    "spo2氧饱和度小": "spo2_min",
    "temperature体温大": "temperature_max",
}

MIMIC_III_RENAME = {
    "lactate_min": "lactate_min",
    "sysbp_min": "sbp_ni_min",
    "spo2_max": "spo2_max",
    "resprate_min": "resp_rate_min",
    "diasbp_max": "dbp_ni_max",
    "heartrate_mean": "heart_rate_mean",
    "spo2_min": "spo2_min",
    "tempc_max": "temperature_max",
}


@dataclass
class Cohort:
    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame
    raw_features: pd.DataFrame


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=script_path.parent / "data",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_path.parent / "analysis_results",
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def set_style() -> None:
    sns.set_theme(style="white")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_header(ax: plt.Axes, label: str, title: str) -> None:
    """Add aligned, non-overlapping uppercase panel labels and short headings."""
    ax.text(
        0.0,
        1.045,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        0.095,
        1.045,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        clip_on=False,
    )


def clean_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = frame[FEATURES].apply(pd.to_numeric, errors="coerce").copy()
    audit_rows = []
    for feature, (low, high) in VALID_RANGES.items():
        series = cleaned[feature]
        invalid = series.notna() & ~series.between(low, high)
        audit_rows.append(
            {
                "feature": feature,
                "display_name": DISPLAY_NAMES[feature],
                "unit": UNITS[feature],
                "valid_min": low,
                "valid_max": high,
                "n_total": len(series),
                "missing_before": int(series.isna().sum()),
                "invalid_set_missing": int(invalid.sum()),
                "minimum_before": float(series.min(skipna=True)),
                "maximum_before": float(series.max(skipna=True)),
            }
        )
        cleaned.loc[invalid, feature] = np.nan
    audit = pd.DataFrame(audit_rows)
    audit["missing_after"] = [
        int(cleaned[row.feature].isna().sum()) for row in audit.itertuples()
    ]
    audit["missing_after_percent"] = 100 * audit["missing_after"] / audit["n_total"]
    return cleaned, audit


def load_mimic_iv(data_root: Path) -> tuple[Cohort, pd.DataFrame, pd.DataFrame]:
    first = pd.read_csv(data_root / "MIMIC IV" / "MIMIC_01.csv")
    second = pd.read_csv(data_root / "MIMIC IV" / "MIMIC_02.csv")
    keys = ["subject_id", "stay_id", "hadm_id"]
    if first[keys].duplicated().any() or second[keys].duplicated().any():
        raise ValueError("MIMIC-IV supplied files are not one row per stay.")
    merged = first.merge(second, on=keys, how="inner", validate="one_to_one")
    if merged["subject_id"].duplicated().any():
        raise ValueError("Repeated patients remain in the MIMIC-IV cohort.")
    original_n = len(merged)
    survival_from_icu = pd.to_numeric(
        merged["icu_survival_day入ICU后存活天数"], errors="coerce"
    )
    alive_at_landmark = survival_from_icu.isna() | survival_from_icu.ge(1.0)
    merged = merged.loc[alive_at_landmark].reset_index(drop=True)

    raw = merged[list(MIMIC_IV_RENAME)].rename(columns=MIMIC_IV_RENAME)
    X, audit = clean_features(raw)
    urine = pd.to_numeric(raw["urine_output"], errors="coerce")
    urine_invalid = urine.notna() & ~urine.between(0, 20000)
    audit = pd.concat(
        [
            audit,
            pd.DataFrame(
                [
                    {
                        "feature": "urine_output",
                        "display_name": DISPLAY_NAMES["urine_output"],
                        "unit": UNITS["urine_output"],
                        "valid_min": 0.0,
                        "valid_max": 20000.0,
                        "n_total": len(urine),
                        "missing_before": int(urine.isna().sum()),
                        "invalid_set_missing": int(urine_invalid.sum()),
                        "minimum_before": float(urine.min(skipna=True)),
                        "maximum_before": float(urine.max(skipna=True)),
                        "missing_after": int((urine.isna() | urine_invalid).sum()),
                        "missing_after_percent": (
                            100 * (urine.isna() | urine_invalid).mean()
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    outcome_col = "death_within_hosp_30days入院30天是否死亡（1为死亡，0为存活）"
    y = pd.to_numeric(merged[outcome_col], errors="raise").astype(int)
    raw_race = merged["race种族"].fillna("UNKNOWN").astype(str).str.upper()
    harmonized_race = np.select(
        [
            raw_race.str.contains("WHITE", regex=False),
            raw_race.str.contains("BLACK", regex=False),
            raw_race.str.contains("HISPANIC", regex=False),
            raw_race.str.contains("UNKNOWN|UNABLE|DECLINED", regex=True),
        ],
        ["White", "Black", "Hispanic", "Unknown"],
        default="Other",
    )
    metadata = pd.DataFrame(
        {
            "subject_id": merged["subject_id"],
            "stay_id": merged["stay_id"],
            "hadm_id": merged["hadm_id"],
            "age": pd.to_numeric(merged["age年龄"], errors="coerce"),
            "sex": merged["gender性别（F女，M男）"].astype(str),
            "race": harmonized_race,
            "sofa": pd.to_numeric(merged["sofa"], errors="coerce"),
            "sapsii": pd.to_numeric(merged["sapsii"], errors="coerce"),
            "outcome": y,
        }
    )
    metadata["missing_feature_count"] = X.isna().sum(axis=1)
    metadata.attrs["landmark_excluded"] = original_n - len(merged)
    return Cohort(X=X, y=y, metadata=metadata, raw_features=raw), merged, audit


def load_mimic_iii(data_root: Path) -> tuple[Cohort, pd.DataFrame, pd.DataFrame]:
    source_path = data_root / "MIMIC.csv"
    observed_source_sha256 = sha256(source_path)
    known_source_hashes = {
        MIMIC_III_SOURCE_SHA256,
        MIMIC_III_ANALYTIC_SHA256,
    }
    if observed_source_sha256 not in known_source_hashes:
        warnings.warn(
            "MIMIC-III file checksum differs from both the cited source supplement "
            "and the reported R2 analytic derivative; record the observed checksum "
            "before interpreting a rerun.",
            RuntimeWarning,
        )
    raw_frame = pd.read_csv(source_path)
    original_n = len(raw_frame)
    if "icu_los" not in raw_frame:
        raise ValueError("MIMIC-III validation data must contain icu_los.")
    icu_los_days = pd.to_numeric(raw_frame["icu_los"], errors="coerce")
    at_24h_icu_landmark = icu_los_days.ge(1.0)
    raw_frame = raw_frame.loc[at_24h_icu_landmark].reset_index(drop=True)
    raw = pd.DataFrame(index=raw_frame.index)
    for source, target in MIMIC_III_RENAME.items():
        raw[target] = raw_frame[source]
    X, audit = clean_features(raw)
    y = pd.to_numeric(raw_frame["thirtyday_expire_flag"], errors="raise").astype(int)
    sex = np.where(pd.to_numeric(raw_frame["is_male"], errors="coerce") == 1, "M", "F")
    race = np.select(
        [
            raw_frame["race_white"].eq(1),
            raw_frame["race_black"].eq(1),
            raw_frame["race_hispanic"].eq(1),
        ],
        ["White", "Black", "Hispanic"],
        default="Other",
    )
    metadata = pd.DataFrame(
        {
            "subject_id": np.arange(len(raw_frame)),
            "age": pd.to_numeric(raw_frame["age"], errors="coerce"),
            "sex": sex,
            "race": race,
            "sofa": pd.to_numeric(raw_frame["sofa"], errors="coerce"),
            "sapsii": np.nan,
            "outcome": y,
        }
    )
    metadata["missing_feature_count"] = X.isna().sum(axis=1)
    metadata.attrs["landmark_excluded"] = original_n - len(raw_frame)
    metadata.attrs["source_sha256"] = observed_source_sha256
    return Cohort(X=X, y=y, metadata=metadata, raw_features=raw), raw_frame, audit


def format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def continuous_row(
    label: str, values: pd.Series, outcome: pd.Series
) -> dict[str, str | int]:
    numeric = pd.to_numeric(values, errors="coerce")
    survivors = numeric[outcome.eq(0)]
    deaths = numeric[outcome.eq(1)]
    p_value = ttest_ind(
        survivors.dropna(), deaths.dropna(), equal_var=False, nan_policy="omit"
    ).pvalue

    def fmt(series: pd.Series) -> str:
        return f"{series.mean():.2f} ± {series.std(ddof=1):.2f}"

    return {
        "Characteristic": label,
        "Overall": fmt(numeric),
        "Survivors": fmt(survivors),
        "Non-survivors": fmt(deaths),
        "p-value": format_p_value(float(p_value)),
        "Missing, n (%)": f"{numeric.isna().sum()} ({100*numeric.isna().mean():.1f}%)",
    }


def categorical_rows(
    label: str, values: pd.Series, outcome: pd.Series
) -> list[dict[str, str]]:
    cat = values.fillna("Missing").astype(str)
    contingency = pd.crosstab(cat, outcome)
    p_value = chi2_contingency(contingency).pvalue if contingency.shape[0] > 1 else np.nan
    rows = []
    for level in sorted(cat.unique()):
        mask = cat.eq(level)
        rows.append(
            {
                "Characteristic": f"{label}: {level}",
                "Overall": f"{mask.sum()} ({100*mask.mean():.1f}%)",
                "Survivors": (
                    f"{(mask & outcome.eq(0)).sum()} "
                    f"({100*(mask & outcome.eq(0)).sum()/max(outcome.eq(0).sum(),1):.1f}%)"
                ),
                "Non-survivors": (
                    f"{(mask & outcome.eq(1)).sum()} "
                    f"({100*(mask & outcome.eq(1)).sum()/max(outcome.eq(1).sum(),1):.1f}%)"
                ),
                "p-value": format_p_value(float(p_value)) if not rows else "",
                "Missing, n (%)": "",
            }
        )
    return rows


def make_baseline_tables(
    merged_iv: pd.DataFrame,
    raw_iii: pd.DataFrame,
    iv: Cohort,
    iii: Cohort,
    tables_dir: Path,
) -> None:
    # Clean the reviewer-flagged descriptive variables before summarization.
    iv_mbp = pd.to_numeric(merged_iv["vitalnbpm无创血压平均值"], errors="coerce")
    iv_resp = pd.to_numeric(merged_iv["vitalrr呼吸"], errors="coerce")
    iv_spo2 = pd.to_numeric(merged_iv["vitalspo2氧饱和度"], errors="coerce")
    iv_mbp = iv_mbp.where(iv_mbp.between(20, 250))
    iv_resp = iv_resp.where(iv_resp.between(1, 80))
    iv_spo2 = iv_spo2.where(iv_spo2.between(30, 100))

    iv_rows = [
        continuous_row("Age, years", iv.metadata["age"], iv.y),
        *categorical_rows("Sex", iv.metadata["sex"], iv.y),
        *categorical_rows("Race", iv.metadata["race"], iv.y),
        continuous_row("Mean arterial pressure, mmHg", iv_mbp, iv.y),
        continuous_row("Mean respiratory rate, breaths/min", iv_resp, iv.y),
        continuous_row("Mean SpO2, %", iv_spo2, iv.y),
        continuous_row(
            "Urine output, mL/24 h",
            pd.to_numeric(merged_iv["尿量(单位ml）"], errors="coerce").where(
                pd.to_numeric(merged_iv["尿量(单位ml）"], errors="coerce").between(
                    0, 20000
                )
            ),
            iv.y,
        ),
        continuous_row("Minimum lactate, mmol/L", iv.X["lactate_min"], iv.y),
        continuous_row("SOFA score", iv.metadata["sofa"], iv.y),
        continuous_row("SAPS II score", iv.metadata["sapsii"], iv.y),
    ]
    pd.DataFrame(iv_rows).to_csv(tables_dir / "Table_1A_MIMIC_IV.csv", index=False)

    glucose = pd.to_numeric(raw_iii["glucose_max1"], errors="coerce")
    glucose = glucose.where(glucose.between(20, 2000))
    iii_rows = [
        continuous_row("Age, years", iii.metadata["age"], iii.y),
        *categorical_rows("Sex", iii.metadata["sex"], iii.y),
        *categorical_rows("Race", iii.metadata["race"], iii.y),
        continuous_row("Maximum glucose, mg/dL", glucose, iii.y),
        continuous_row("Mean respiratory rate, breaths/min", raw_iii["resprate_mean"], iii.y),
        continuous_row("Mean SpO2, %", raw_iii["spo2_mean"], iii.y),
        continuous_row("Minimum lactate, mmol/L", iii.X["lactate_min"], iii.y),
        continuous_row("SOFA score", iii.metadata["sofa"], iii.y),
    ]
    pd.DataFrame(iii_rows).to_csv(tables_dir / "Table_1B_MIMIC_III.csv", index=False)


def build_model_specs(quick: bool) -> dict[str, tuple[Pipeline, dict]]:
    folds = 3
    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    random_state=SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    svm = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    probability=True,
                    random_state=SEED,
                    class_weight="balanced",
                    cache_size=2000,
                ),
            ),
        ]
    )
    xgb = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                xgboost.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=SEED,
                    n_jobs=4,
                    tree_method="hist",
                ),
            ),
        ]
    )
    lightgbm = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                LGBMClassifier(
                    objective="binary",
                    random_state=SEED,
                    n_jobs=4,
                    verbosity=-1,
                ),
            ),
        ]
    )
    adaboost = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", AdaBoostClassifier(random_state=SEED)),
        ]
    )
    catboost = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                CatBoostClassifier(
                    random_seed=SEED,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=4,
                ),
            ),
        ]
    )
    if quick:
        return {
            "Logistic regression": (logistic, {"classifier__C": [1.0]}),
            "SVM": (svm, {"classifier__C": [1.0], "classifier__gamma": ["scale"]}),
            "XGBoost": (
                xgb,
                {
                    "classifier__n_estimators": [150],
                    "classifier__max_depth": [4],
                    "classifier__learning_rate": [0.05],
                    "classifier__subsample": [0.8],
                    "classifier__colsample_bytree": [1.0],
                },
            ),
            "LightGBM": (
                lightgbm,
                {
                    "classifier__n_estimators": [150],
                    "classifier__num_leaves": [15],
                    "classifier__learning_rate": [0.05],
                },
            ),
            "AdaBoost": (
                adaboost,
                {"classifier__n_estimators": [150], "classifier__learning_rate": [0.05]},
            ),
            "CatBoost": (
                catboost,
                {
                    "classifier__iterations": [200],
                    "classifier__depth": [5],
                    "classifier__learning_rate": [0.05],
                },
            ),
        }
    return {
        "Logistic regression": (logistic, {"classifier__C": [0.1, 1.0, 10.0]}),
        "SVM": (
            svm,
            {"classifier__C": [0.5, 1.0], "classifier__gamma": ["scale"]},
        ),
        "XGBoost": (
            xgb,
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [0.05, 0.10],
                "classifier__subsample": [0.8, 1.0],
                "classifier__colsample_bytree": [0.8, 1.0],
            },
        ),
        "LightGBM": (
            lightgbm,
            {
                "classifier__n_estimators": [100, 200],
                "classifier__num_leaves": [15, 31],
                "classifier__learning_rate": [0.05, 0.10],
            },
        ),
        "AdaBoost": (
            adaboost,
            {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.10],
            },
        ),
        "CatBoost": (
            catboost,
            {
                "classifier__iterations": [150, 250],
                "classifier__depth": [4, 6],
                "classifier__learning_rate": [0.05, 0.10],
            },
        ),
    }


def fit_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    tables_dir: Path,
    figures_dir: Path,
) -> list[str]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    Cs = np.logspace(-3, 2, 24)
    lasso_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegressionCV(
                    Cs=Cs,
                    cv=cv,
                    scoring="roc_auc",
                    solver="liblinear",
                    penalty="l1",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=SEED,
                    refit=True,
                    n_jobs=3,
                ),
            ),
        ]
    )
    lasso_pipe.fit(X, y)
    classifier = lasso_pipe.named_steps["classifier"]
    coefficients = pd.Series(classifier.coef_[0], index=FEATURES, name="coefficient")
    selected = coefficients[coefficients.ne(0)].index.tolist()
    if not selected:
        raise RuntimeError("Logistic LASSO selected no features.")

    imputed = lasso_pipe.named_steps["imputer"].transform(X)
    scaled = lasso_pipe.named_steps["scaler"].transform(imputed)
    selected_indices = [FEATURES.index(f) for f in selected]
    selected_scaled = scaled[:, selected_indices]
    vif = pd.DataFrame(
        {
            "feature": selected,
            "VIF": [
                variance_inflation_factor(selected_scaled, index)
                for index in range(selected_scaled.shape[1])
            ],
        }
    )
    selection_table = (
        coefficients.rename_axis("feature")
        .reset_index()
        .merge(vif, on="feature", how="left")
    )
    selection_table["selected"] = selection_table["coefficient"].ne(0)
    selection_table["C_selected"] = float(classifier.C_[0])
    selection_table.to_csv(tables_dir / "Table_S2_feature_selection.csv", index=False)

    # Fit the coefficient path on the already training-fitted transformation.
    path_rows = []
    for C in Cs:
        model = LogisticRegression(
            C=float(C),
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        )
        model.fit(scaled, y)
        for feature, value in zip(FEATURES, model.coef_[0]):
            path_rows.append({"C": C, "feature": feature, "coefficient": value})
    path = pd.DataFrame(path_rows)

    scores = classifier.scores_[1]
    mean_auc = scores.mean(axis=0)
    se_auc = scores.std(axis=0, ddof=1) / np.sqrt(scores.shape[0])
    cv_table = pd.DataFrame({"C": Cs, "mean_AUROC": mean_auc, "SE": se_auc})
    cv_table.to_csv(tables_dir / "Table_S3_lasso_cv.csv", index=False)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.8),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.25]},
    )
    path_colors = [
        FIGURE_PALETTE["navy"],
        FIGURE_PALETTE["gold"],
        FIGURE_PALETTE["periwinkle"],
        FIGURE_PALETTE["green"],
    ]
    path_styles = ["-", "--", ":", "-."]
    for feature_index, feature in enumerate(FEATURES):
        subset = path[path["feature"].eq(feature)]
        axes[0].plot(
            np.log10(subset["C"]),
            subset["coefficient"],
            color=path_colors[feature_index % len(path_colors)],
            linestyle=path_styles[(feature_index // len(path_colors)) % len(path_styles)],
            lw=1.15,
        )
    axes[0].axvline(
        np.log10(float(classifier.C_[0])),
        color=FIGURE_PALETTE["green"],
        ls="--",
        lw=1,
    )
    axes[0].set_xlabel("log10(C)")
    axes[0].set_ylabel("L1 coefficient")
    panel_header(axes[0], "A", "Coefficient paths")

    axes[1].errorbar(
        np.log10(Cs),
        mean_auc,
        yerr=se_auc,
        fmt="o-",
        color=FIGURE_PALETTE["navy"],
        ecolor=FIGURE_PALETTE["periwinkle"],
        ms=3,
        lw=1,
        capsize=2,
    )
    axes[1].axvline(
        np.log10(float(classifier.C_[0])),
        color=FIGURE_PALETTE["green"],
        ls="--",
        lw=1,
    )
    axes[1].set_xlabel("log10(C)")
    axes[1].set_ylabel("Cross-validated AUROC")
    panel_header(axes[1], "B", "Cross-validation")

    ordered = coefficients[coefficients.ne(0)].sort_values()
    feature_y = np.arange(len(ordered))
    axes[2].barh(
        feature_y,
        ordered.values,
        color=[
            FIGURE_PALETTE["gold"]
            if value > 0
            else FIGURE_PALETTE["navy"]
            for value in ordered
        ],
        edgecolor="white",
        linewidth=0.4,
    )
    axes[2].set_yticks(feature_y)
    axes[2].set_yticklabels([])
    axes[2].tick_params(axis="y", length=0)
    axes[2].spines["left"].set_visible(False)
    axes[2].set_xlim(-0.43, 0.42)
    for y_position, feature in zip(feature_y, ordered.index):
        axes[2].text(
            -0.21,
            y_position,
            SHORT_NAMES[feature],
            ha="right",
            va="center",
            fontsize=7,
        )
    axes[2].axvline(0, color=FIGURE_PALETTE["green"], lw=0.8)
    axes[2].set_xlabel("Standardized coefficient")
    panel_header(axes[2], "C", "Selected features")
    for ax in axes[:2]:
        ax.grid(axis="y", color=FIGURE_PALETTE["cream"], linewidth=0.7)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.82, bottom=0.20, wspace=0.34)
    save_figure(fig, figures_dir / "Figure_2_Feature_selection")
    plt.close(fig)
    return selected


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    eligible = np.flatnonzero(tpr >= TARGET_SENSITIVITY)
    if eligible.size == 0:
        return 0.5
    best_index = eligible[np.argmin(fpr[eligible])]
    threshold = float(thresholds[best_index])
    return min(max(threshold, 0.001), 0.999)


def calibration_parameters(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    eps = 1e-6
    logits = np.log(np.clip(probabilities, eps, 1 - eps) / np.clip(1 - probabilities, eps, 1 - eps))
    design = sm.add_constant(logits)
    try:
        fit = sm.Logit(y_true, design).fit(disp=0)
        return float(fit.params[0]), float(fit.params[1])
    except Exception:
        return np.nan, np.nan


def point_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    intercept, slope = calibration_parameters(y_true, probabilities)
    return {
        "AUROC": roc_auc_score(y_true, probabilities),
        "PR_AUC": average_precision_score(y_true, probabilities),
        "Brier": brier_score_loss(y_true, probabilities),
        "Calibration_intercept": intercept,
        "Calibration_slope": slope,
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "Sensitivity": recall_score(y_true, predictions, zero_division=0),
        "Specificity": tn / max(tn + fp, 1),
        "PPV": precision_score(y_true, predictions, zero_division=0),
        "NPV": tn / max(tn + fn, 1),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Threshold": threshold,
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    n_bootstrap: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, n)
        sampled_y = y_true[indices]
        if np.unique(sampled_y).size < 2:
            continue
        rows.append(point_metrics(sampled_y, probabilities[indices], threshold))
    return pd.DataFrame(rows)


def summarize_with_ci(
    dataset: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    n_bootstrap: int,
) -> pd.DataFrame:
    point = point_metrics(y_true, probabilities, threshold)
    boot = bootstrap_metrics(y_true, probabilities, threshold, n_bootstrap)
    rows = []
    for metric, estimate in point.items():
        if metric in {"TN", "FP", "FN", "TP", "Threshold"}:
            rows.append(
                {
                    "Dataset": dataset,
                    "Metric": metric,
                    "Estimate": estimate,
                    "CI_low": np.nan,
                    "CI_high": np.nan,
                }
            )
            continue
        rows.append(
            {
                "Dataset": dataset,
                "Metric": metric,
                "Estimate": estimate,
                "CI_low": boot[metric].quantile(0.025),
                "CI_high": boot[metric].quantile(0.975),
            }
        )
    return pd.DataFrame(rows)


def subgroup_table(
    dataset: str,
    y: pd.Series,
    probabilities: np.ndarray,
    metadata: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    groups = {
        "Age": pd.cut(
            metadata["age"],
            bins=[0, 64, 79, np.inf],
            labels=["<65", "65-79", ">=80"],
        ).astype(str),
        "Sex": metadata["sex"].astype(str),
        "Race": metadata["race"].astype(str),
        "SOFA": pd.cut(
            metadata["sofa"],
            bins=[-np.inf, 5, 9, np.inf],
            labels=["<6", "6-9", ">=10"],
        ).astype(str),
        "Missing features": pd.cut(
            metadata["missing_feature_count"],
            bins=[-1, 0, 1, np.inf],
            labels=["0", "1", ">=2"],
        ).astype(str),
    }
    rows = []
    y_array = y.to_numpy()
    rng = np.random.default_rng(SEED)
    for grouping, values in groups.items():
        for level in sorted(pd.Series(values).dropna().unique()):
            mask = pd.Series(values).eq(level).to_numpy()
            if mask.sum() < 30 or np.unique(y_array[mask]).size < 2:
                continue
            subgroup_y = y_array[mask]
            subgroup_probabilities = probabilities[mask]
            metrics = point_metrics(subgroup_y, subgroup_probabilities, threshold)
            bootstrap_aurocs = []
            bootstrap_sensitivities = []
            for _ in range(N_BOOTSTRAP):
                indices = rng.integers(0, len(subgroup_y), len(subgroup_y))
                sampled_y = subgroup_y[indices]
                if np.unique(sampled_y).size < 2:
                    continue
                sampled_probabilities = subgroup_probabilities[indices]
                bootstrap_aurocs.append(
                    roc_auc_score(sampled_y, sampled_probabilities)
                )
                sampled_predictions = sampled_probabilities >= threshold
                bootstrap_sensitivities.append(
                    recall_score(
                        sampled_y,
                        sampled_predictions,
                        zero_division=0,
                    )
                )
            rows.append(
                {
                    "Dataset": dataset,
                    "Grouping": grouping,
                    "Level": level,
                    "N": int(mask.sum()),
                    "Events": int(y_array[mask].sum()),
                    "AUROC": metrics["AUROC"],
                    "PR_AUC": metrics["PR_AUC"],
                    "Sensitivity": metrics["Sensitivity"],
                    "Specificity": metrics["Specificity"],
                    "Brier": metrics["Brier"],
                    "Calibration_intercept": metrics["Calibration_intercept"],
                    "Calibration_slope": metrics["Calibration_slope"],
                    "AUROC_CI_low": np.quantile(bootstrap_aurocs, 0.025),
                    "AUROC_CI_high": np.quantile(bootstrap_aurocs, 0.975),
                    "Sensitivity_CI_low": np.quantile(
                        bootstrap_sensitivities, 0.025
                    ),
                    "Sensitivity_CI_high": np.quantile(
                        bootstrap_sensitivities, 0.975
                    ),
                }
            )
    return pd.DataFrame(rows)


def decision_curve(
    y_true: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> pd.DataFrame:
    rows = []
    n = len(y_true)
    prevalence = y_true.mean()
    for threshold in thresholds:
        predictions = probabilities >= threshold
        fp = np.sum(predictions & (y_true == 0))
        tp = np.sum(predictions & (y_true == 1))
        odds = threshold / (1 - threshold)
        rows.append(
            {
                "threshold": threshold,
                "model": tp / n - fp / n * odds,
                "treat_all": prevalence - (1 - prevalence) * odds,
                "treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def save_figure(
    fig: plt.Figure,
    stem: Path,
    *,
    tight: bool = True,
) -> None:
    bbox_inches = "tight" if tight else None
    fig.savefig(stem.with_suffix(".svg"), bbox_inches=bbox_inches, pad_inches=0.04)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches=bbox_inches, pad_inches=0.04)
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches=bbox_inches,
        pad_inches=0.04,
    )
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches=bbox_inches,
        pad_inches=0.04,
    )


def plot_discrimination_summary(
    y_internal: np.ndarray,
    p_internal: np.ndarray,
    y_external: np.ndarray,
    p_external: np.ndarray,
    tuning: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Show development selection separately from locked-cohort discrimination."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.25))

    tuning = tuning.sort_values("Best_CV_AUROC").reset_index(drop=True)
    y_positions = np.arange(len(tuning))
    colors = [MODEL_STYLES[name]["color"] for name in tuning["Model"]]
    markers = [MODEL_STYLES[name]["marker"] for name in tuning["Model"]]
    axes[0].hlines(
        y_positions,
        0.65,
        tuning["Best_CV_AUROC"],
        color=FIGURE_PALETTE["cream"],
        linewidth=2.2,
        zorder=1,
    )
    for y_position, value, color, marker in zip(
        y_positions,
        tuning["Best_CV_AUROC"],
        colors,
        markers,
    ):
        axes[0].scatter(
            value,
            y_position,
            color=color,
            marker=marker,
            s=30,
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        axes[0].text(
            value + 0.001,
            y_position,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=6.2,
        )
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(tuning["Model"], fontsize=6.4)
    axes[0].set_xlim(0.65, 0.705)
    axes[0].set_xlabel("Mean cross-validated AUROC")
    axes[0].grid(axis="x", color=FIGURE_PALETTE["cream"], linewidth=0.7)
    panel_header(axes[0], "A", "Model selection")

    evaluation_sets = [
        (
            "Internal test",
            y_internal,
            p_internal,
            FIGURE_PALETTE["navy"],
            "-",
        ),
        (
            "MIMIC-III",
            y_external,
            p_external,
            FIGURE_PALETTE["gold"],
            "--",
        ),
    ]
    for label, y_true, probabilities, color, linestyle in evaluation_sets:
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        axes[1].plot(
            fpr,
            tpr,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            label=f"{label} ({roc_auc_score(y_true, probabilities):.3f})",
        )
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        axes[2].plot(
            recall,
            precision,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            label=(
                f"{label} "
                f"({average_precision_score(y_true, probabilities):.3f})"
            ),
        )
    axes[1].plot(
        [0, 1],
        [0, 1],
        color=FIGURE_PALETTE["periwinkle"],
        linestyle=":",
        linewidth=1.0,
    )
    axes[1].set_xlabel("False-positive rate")
    axes[1].set_ylabel("True-positive rate")
    axes[1].legend(frameon=False, fontsize=6.2, loc="lower right")
    panel_header(axes[1], "B", "ROC curves")

    for _, y_true, _, color, linestyle in evaluation_sets:
        axes[2].axhline(
            y_true.mean(),
            color=color,
            linestyle=":",
            linewidth=0.8,
            alpha=0.7,
        )
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].set_ylim(0, 1)
    axes[2].legend(frameon=False, fontsize=6.2, loc="upper right")
    panel_header(axes[2], "C", "Precision–recall curves")
    for ax in axes[1:]:
        ax.set_xlim(0, 1)
        ax.grid(color=FIGURE_PALETTE["cream"], linewidth=0.7)
    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        top=0.82,
        bottom=0.18,
        wspace=0.43,
    )
    save_figure(fig, figures_dir / "Figure_3_Model_performance")
    plt.close(fig)


def plot_model_performance(
    model_probabilities: dict[str, np.ndarray],
    y_test: pd.Series,
    y_external: pd.Series,
    p_internal: np.ndarray,
    p_external: np.ndarray,
    threshold: float,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    rows = []
    for name, probabilities in model_probabilities.items():
        if name == "SepXAI":
            continue
        metrics = point_metrics(y_test.to_numpy(), probabilities, 0.5)
        rows.append({"Model": name, **metrics})
    performance = pd.DataFrame(rows)
    performance.to_csv(tables_dir / "Table_S4_model_comparison.csv", index=False)
    plot_discrimination_summary(
        y_test.to_numpy(),
        p_internal,
        y_external.to_numpy(),
        p_external,
        pd.read_csv(tables_dir / "Table_S13_hyperparameter_tuning.csv"),
        figures_dir,
    )


def plot_calibration_and_dca(
    y_internal: pd.Series,
    p_internal: np.ndarray,
    y_external: pd.Series,
    p_external: np.ndarray,
    operating_threshold: float,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    thresholds = np.linspace(0.05, 0.50, 91)
    dca_internal = decision_curve(y_internal.to_numpy(), p_internal, thresholds)
    dca_external = decision_curve(y_external.to_numpy(), p_external, thresholds)
    dca_internal.assign(dataset="Internal test").to_csv(
        tables_dir / "Table_S5_DCA_internal.csv", index=False
    )
    dca_external.assign(dataset="External validation").to_csv(
        tables_dir / "Table_S6_DCA_external.csv", index=False
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.35))
    for ax, label, y, probabilities in [
        (axes[0, 0], ("A", "Internal calibration"), y_internal, p_internal),
        (axes[0, 1], ("B", "External calibration"), y_external, p_external),
    ]:
        observed, predicted = calibration_curve(y, probabilities, n_bins=10, strategy="quantile")
        ax.plot(
            predicted,
            observed,
            "o-",
            color=FIGURE_PALETTE["navy"],
            markerfacecolor=FIGURE_PALETTE["gold"],
            markeredgecolor=FIGURE_PALETTE["navy"],
            lw=1.4,
            ms=3.5,
        )
        ax.plot(
            [0, 1],
            [0, 1],
            "--",
            color=FIGURE_PALETTE["periwinkle"],
            lw=0.9,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted risk")
        ax.set_ylabel("Observed mortality")
        ax.grid(color=FIGURE_PALETTE["cream"], linewidth=0.7)
        panel_header(ax, *label)
    for ax, label, dca in [
        (axes[1, 0], ("C", "Internal decision curve"), dca_internal),
        (axes[1, 1], ("D", "External decision curve"), dca_external),
    ]:
        ax.plot(
            dca["threshold"],
            dca["model"],
            color=FIGURE_PALETTE["navy"],
            lw=1.5,
            label="SepXAI",
        )
        ax.plot(
            dca["threshold"],
            dca["treat_all"],
            color=FIGURE_PALETTE["gold"],
            ls="--",
            lw=1.1,
            label="Treat all",
        )
        ax.plot(
            dca["threshold"],
            dca["treat_none"],
            color=FIGURE_PALETTE["green"],
            ls=":",
            lw=1.1,
            label="Treat none",
        )
        ax.axvline(
            operating_threshold,
            color=FIGURE_PALETTE["periwinkle"],
            ls="-.",
            lw=1,
            label="Operating threshold",
        )
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.grid(axis="y", color=FIGURE_PALETTE["cream"], linewidth=0.7)
        panel_header(ax, *label)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        fontsize=6.5,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.14, hspace=0.48, wspace=0.28)
    save_figure(fig, figures_dir / "Figure_4_Calibration_DCA")
    plt.close(fig)


def plot_shap_and_lime(
    fitted_base_pipeline: Pipeline,
    X_training: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    p_test: np.ndarray,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    feature_names = list(X_training.columns)
    imputer = fitted_base_pipeline.named_steps["imputer"]
    classifier = fitted_base_pipeline.named_steps["classifier"]
    sample = X_test.sample(min(1200, len(X_test)), random_state=SEED)
    sample_imputed = imputer.transform(sample)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(sample_imputed)
    shap_frame = pd.DataFrame(
        shap_values.values, columns=feature_names, index=sample.index
    )
    shap_frame.to_csv(tables_dir / "Source_data_Figure_6_SHAP.csv", index=True)

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    global_order = np.argsort(mean_abs)[::-1]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.65),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    shap_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "sepxai_feature_value",
        [
            FIGURE_PALETTE["navy"],
            FIGURE_PALETTE["cream"],
            FIGURE_PALETTE["gold"],
        ],
    )
    rng = np.random.default_rng(SEED)
    scatter_handle = None
    for y_position, feature_index in enumerate(global_order):
        feature_values = sample_imputed[:, feature_index]
        ranks = pd.Series(feature_values).rank(method="average", pct=True).to_numpy()
        jitter = rng.normal(0, 0.085, size=len(feature_values))
        scatter_handle = axes[0].scatter(
            shap_values.values[:, feature_index],
            y_position + jitter,
            c=ranks,
            cmap=shap_cmap,
            vmin=0,
            vmax=1,
            s=7,
            alpha=0.68,
            linewidth=0,
        )
    axes[0].axvline(0, color=FIGURE_PALETTE["green"], lw=0.8)
    axes[0].set_yticks(np.arange(len(global_order)))
    axes[0].set_yticklabels([SHORT_NAMES[feature_names[i]] for i in global_order])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("SHAP value")
    axes[0].grid(axis="x", color=FIGURE_PALETTE["cream"], linewidth=0.7)
    panel_header(axes[0], "A", "SHAP distribution")

    axes[1].barh(
        np.arange(len(global_order)),
        mean_abs[global_order],
        color=FIGURE_PALETTE["navy"],
        edgecolor="white",
        linewidth=0.4,
    )
    axes[1].set_yticks(np.arange(len(global_order)))
    axes[1].set_yticklabels([SHORT_NAMES[feature_names[i]] for i in global_order])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean absolute SHAP value")
    axes[1].grid(axis="x", color=FIGURE_PALETTE["cream"], linewidth=0.7)
    panel_header(axes[1], "B", "Global importance")
    fig.subplots_adjust(left=0.16, right=0.985, top=0.88, bottom=0.22, wspace=0.60)
    panel_position = axes[0].get_position()
    color_axis = fig.add_axes([panel_position.x0, 0.055, panel_position.width, 0.026])
    color_steps = 64
    for step in range(color_steps):
        color_axis.add_patch(
            mpl.patches.Rectangle(
                (step / color_steps, 0),
                1 / color_steps,
                1,
                facecolor=shap_cmap((step + 0.5) / color_steps),
                edgecolor="none",
            )
        )
    color_axis.set_xlim(0, 1)
    color_axis.set_ylim(0, 1)
    color_axis.set_yticks([])
    color_axis.set_xticks([0, 1])
    color_axis.set_xticklabels(["Low", "High"])
    color_axis.set_xlabel("Feature value", labelpad=-1)
    color_axis.tick_params(axis="x", length=0, pad=3)
    for spine in color_axis.spines.values():
        spine.set_linewidth(0.7)
    save_figure(fig, figures_dir / "Figure_6_Global_SHAP")
    plt.close(fig)

    high_risk_index = X_test.index[int(np.argmax(p_test))]
    high_risk_row = X_test.loc[[high_risk_index]]
    high_imputed = imputer.transform(high_risk_row)
    high_shap = explainer(high_imputed)
    training_imputed = imputer.transform(X_training)
    lime_explainer = LimeTabularExplainer(
        training_imputed,
        feature_names=[DISPLAY_NAMES[f] for f in feature_names],
        class_names=["Survival", "30-day mortality"],
        discretize_continuous=True,
        random_state=SEED,
    )
    lime_result = lime_explainer.explain_instance(
        high_imputed[0],
        classifier.predict_proba,
        num_features=len(feature_names),
    )
    lime_table = pd.DataFrame(lime_result.as_list(), columns=["Rule", "Weight"])
    lime_table.to_csv(tables_dir / "Source_data_Figure_7_LIME.csv", index=False)

    fig = plt.figure(figsize=(7.2, 3.7))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15])
    ax1 = fig.add_subplot(grid[0, 0])
    local_shap = high_shap.values[0]
    shap_order = np.argsort(np.abs(local_shap))[::-1]
    shap_colors = [
        FIGURE_PALETTE["gold"]
        if local_shap[index] > 0
        else FIGURE_PALETTE["navy"]
        for index in shap_order
    ]
    ax1.barh(
        np.arange(len(shap_order)),
        local_shap[shap_order],
        color=shap_colors,
        edgecolor="white",
        linewidth=0.4,
    )
    ax1.set_yticks(np.arange(len(shap_order)))
    ax1.set_yticklabels([SHORT_NAMES[feature_names[i]] for i in shap_order])
    ax1.invert_yaxis()
    ax1.axvline(0, color=FIGURE_PALETTE["green"], lw=0.8)
    ax1.set_xlabel("SHAP contribution")
    ax1.grid(axis="x", color=FIGURE_PALETTE["cream"], linewidth=0.7)
    panel_header(ax1, "A", "SHAP")

    ax2 = fig.add_subplot(grid[0, 1])
    compact_rules = lime_table["Rule"].copy()
    lime_rule_names = {
        **SHORT_NAMES,
        "temperature_max": "Temp., max",
        "resp_rate_min": "RR, min",
    }
    for feature in sorted(feature_names, key=lambda item: len(DISPLAY_NAMES[item]), reverse=True):
        compact_rules = compact_rules.str.replace(
            DISPLAY_NAMES[feature],
            lime_rule_names[feature],
            regex=False,
        )
    ordered_lime = lime_table.assign(Compact_rule=compact_rules)
    ordered_lime = ordered_lime.iloc[
        np.argsort(np.abs(ordered_lime["Weight"].to_numpy()))[::-1]
    ]
    ax2.barh(
        np.arange(len(ordered_lime)),
        ordered_lime["Weight"],
        color=[
            FIGURE_PALETTE["gold"]
            if value > 0
            else FIGURE_PALETTE["navy"]
            for value in ordered_lime["Weight"]
        ],
        edgecolor="white",
        linewidth=0.4,
    )
    ax2.set_yticks(np.arange(len(ordered_lime)))
    ax2.set_yticklabels([])
    ax2.tick_params(axis="y", length=0)
    ax2.spines["left"].set_visible(False)
    negative_edge = min(0.0, float(ordered_lime["Weight"].min()))
    label_x = negative_edge - 0.012
    left_limit = label_x - 0.20
    ax2.set_xlim(left_limit, max(0.17, float(ordered_lime["Weight"].max()) * 1.06))
    for y_position, rule in enumerate(ordered_lime["Compact_rule"]):
        ax2.text(
            label_x,
            y_position,
            rule,
            ha="right",
            va="center",
            fontsize=7,
        )
    ax2.invert_yaxis()
    ax2.axvline(0, color=FIGURE_PALETTE["green"], lw=0.8)
    ax2.set_xticks([0.00, 0.05, 0.10, 0.15])
    ax2.set_xlabel("LIME weight")
    ax2.grid(axis="x", color=FIGURE_PALETTE["cream"], linewidth=0.7)
    panel_header(ax2, "B", "LIME")

    legend_handles = [
        mpl.patches.Patch(
            facecolor=FIGURE_PALETTE["gold"],
            edgecolor="none",
            label="Increases model output",
        ),
        mpl.patches.Patch(
            facecolor=FIGURE_PALETTE["navy"],
            edgecolor="none",
            label="Decreases model output",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        fontsize=6.5,
    )
    fig.subplots_adjust(left=0.15, right=0.985, top=0.88, bottom=0.18, wspace=0.34)
    save_figure(fig, figures_dir / "Figure_7_Patient_explanations")
    plt.close(fig)

    patient_record = high_risk_row.copy()
    patient_record["observed_outcome"] = int(y_test.loc[high_risk_index])
    patient_record["predicted_probability"] = float(p_test[X_test.index.get_loc(high_risk_index)])
    patient_record.to_csv(tables_dir / "Figure_7_example_patient.csv", index=False)


def safety_analyses(
    final_model: CalibratedClassifierCV,
    X_training: pd.DataFrame,
    y_calibration: pd.Series,
    p_calibration: np.ndarray,
    X_internal: pd.DataFrame,
    y_internal: pd.Series,
    p_internal: np.ndarray,
    meta_internal: pd.DataFrame,
    X_external: pd.DataFrame,
    y_external: pd.Series,
    p_external: np.ndarray,
    meta_external: pd.DataFrame,
    threshold: float,
    tables_dir: Path,
) -> None:
    feature_names = list(X_training.columns)
    subgroup = pd.concat(
        [
            subgroup_table(
                "Internal test", y_internal, p_internal, meta_internal, threshold
            ),
            subgroup_table(
                "External validation", y_external, p_external, meta_external, threshold
            ),
        ],
        ignore_index=True,
    )
    subgroup.to_csv(tables_dir / "Table_S7_subgroup_performance.csv", index=False)

    quantiles = X_training.quantile([0.01, 0.99])
    ood = pd.DataFrame(index=X_external.index)
    for feature in feature_names:
        ood[feature] = (
            X_external[feature].notna()
            & (
                (X_external[feature] < quantiles.loc[0.01, feature])
                | (X_external[feature] > quantiles.loc[0.99, feature])
            )
        )
    ood_count = ood.sum(axis=1)
    ood_rows = []
    for label, mask in {
        "Within training 1st-99th percentile envelope": ood_count.lt(2),
        "OOD: >=2 features outside envelope": ood_count.ge(2),
    }.items():
        if mask.sum() >= 30 and np.unique(y_external[mask]).size > 1:
            metrics = point_metrics(
                y_external[mask].to_numpy(), p_external[mask.to_numpy()], threshold
            )
            ood_rows.append(
                {
                    "Group": label,
                    "N": int(mask.sum()),
                    "Events": int(y_external[mask].sum()),
                    **metrics,
                }
            )
    pd.DataFrame(ood_rows).to_csv(tables_dir / "Table_S8_OOD_performance.csv", index=False)

    perturbation_rows = []
    baseline = point_metrics(y_internal.to_numpy(), p_internal, threshold)
    for feature in feature_names:
        perturbed = X_internal.copy()
        perturbed[feature] = np.nan
        probabilities = final_model.predict_proba(perturbed)[:, 1]
        metrics = point_metrics(y_internal.to_numpy(), probabilities, threshold)
        perturbation_rows.append(
            {
                "Feature masked": feature,
                "AUROC": metrics["AUROC"],
                "Delta_AUROC": metrics["AUROC"] - baseline["AUROC"],
                "Brier": metrics["Brier"],
                "Delta_Brier": metrics["Brier"] - baseline["Brier"],
                "Sensitivity": metrics["Sensitivity"],
                "Delta_sensitivity": metrics["Sensitivity"] - baseline["Sensitivity"],
            }
        )
    pd.DataFrame(perturbation_rows).to_csv(
        tables_dir / "Table_S9_missing_input_stress_test.csv", index=False
    )

    predictions = p_external >= threshold
    fn_mask = (y_external.to_numpy() == 1) & ~predictions
    tp_mask = (y_external.to_numpy() == 1) & predictions
    false_negative_rows = []
    for feature in feature_names:
        false_negative_rows.append(
            {
                "Characteristic": DISPLAY_NAMES[feature],
                "False negatives": X_external.loc[fn_mask, feature].median(),
                "True positives": X_external.loc[tp_mask, feature].median(),
            }
        )
    for feature in ["age", "sofa", "missing_feature_count"]:
        false_negative_rows.append(
            {
                "Characteristic": feature,
                "False negatives": meta_external.loc[fn_mask, feature].median(),
                "True positives": meta_external.loc[tp_mask, feature].median(),
            }
        )
    pd.DataFrame(false_negative_rows).to_csv(
        tables_dir / "Table_S10_false_negative_audit.csv", index=False
    )

    threshold_rows = []
    candidate_thresholds = sorted(
        set([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, threshold])
    )
    for dataset, y, probabilities in [
        ("Calibration/threshold subset", y_calibration, p_calibration),
        ("Internal test", y_internal, p_internal),
        ("External validation", y_external, p_external),
    ]:
        for candidate in candidate_thresholds:
            threshold_rows.append(
                {
                    "Dataset": dataset,
                    **point_metrics(y.to_numpy(), probabilities, candidate),
                }
            )
    pd.DataFrame(threshold_rows).to_csv(
        tables_dir / "Table_S11_threshold_safety.csv", index=False
    )


def plot_safety_boundaries(
    figures_dir: Path,
    tables_dir: Path,
    threshold: float,
) -> None:
    """Summarize the prespecified safety-boundary analyses in the main text."""
    subgroup = pd.read_csv(tables_dir / "Table_S7_subgroup_performance.csv")
    masking = pd.read_csv(tables_dir / "Table_S9_missing_input_stress_test.csv")
    threshold_table = pd.read_csv(tables_dir / "Table_S11_threshold_safety.csv")

    external_thresholds = (
        threshold_table.loc[
            threshold_table["Dataset"].eq("External validation"),
            ["Threshold", "FN", "FP"],
        ]
        .sort_values("Threshold")
        .reset_index(drop=True)
    )
    missingness = subgroup.loc[
        subgroup["Dataset"].eq("Internal test")
        & subgroup["Grouping"].eq("Missing features"),
        [
            "Level",
            "N",
            "Sensitivity",
            "Sensitivity_CI_low",
            "Sensitivity_CI_high",
        ],
    ].copy()
    missing_order = {"0": 0, "1": 1, ">=2": 2}
    missingness["_order"] = missingness["Level"].astype(str).map(missing_order)
    missingness = missingness.sort_values("_order").drop(columns="_order")

    masking = masking.sort_values("Delta_sensitivity").reset_index(drop=True)
    age = subgroup.loc[
        subgroup["Grouping"].eq("Age"),
        [
            "Dataset",
            "Level",
            "N",
            "AUROC",
            "AUROC_CI_low",
            "AUROC_CI_high",
        ],
    ].copy()
    age_order = ["<65", "65-79", ">=80"]
    age["Level"] = pd.Categorical(age["Level"], categories=age_order, ordered=True)
    age = age.sort_values(["Level", "Dataset"]).reset_index(drop=True)

    source_rows: list[dict[str, object]] = []
    for row in external_thresholds.itertuples(index=False):
        source_rows.extend(
            [
                {
                    "Panel": "A",
                    "Dataset": "External validation",
                    "Group": f"Threshold {row.Threshold:.6f}",
                    "Metric": "False negatives",
                    "Value": int(row.FN),
                    "N": np.nan,
                },
                {
                    "Panel": "A",
                    "Dataset": "External validation",
                    "Group": f"Threshold {row.Threshold:.6f}",
                    "Metric": "False positives",
                    "Value": int(row.FP),
                    "N": np.nan,
                },
            ]
        )
    for row in missingness.itertuples(index=False):
        source_rows.append(
            {
                "Panel": "B",
                "Dataset": "Internal test",
                "Group": str(row.Level),
                "Metric": "Sensitivity",
                "Value": row.Sensitivity,
                "N": int(row.N),
                "CI_low": row.Sensitivity_CI_low,
                "CI_high": row.Sensitivity_CI_high,
            }
        )
    for _, row in masking.iterrows():
        source_rows.append(
            {
                "Panel": "C",
                "Dataset": "Internal test",
                "Group": row["Feature masked"],
                "Metric": "Change in sensitivity",
                "Value": row["Delta_sensitivity"],
                "N": np.nan,
            }
        )
    for row in age.itertuples(index=False):
        source_rows.append(
            {
                "Panel": "D",
                "Dataset": row.Dataset,
                "Group": str(row.Level),
                "Metric": "AUROC",
                "Value": row.AUROC,
                "N": int(row.N),
                "CI_low": row.AUROC_CI_low,
                "CI_high": row.AUROC_CI_high,
            }
        )
    source_data = pd.DataFrame(source_rows)
    source_data["Locked_threshold"] = threshold
    source_data.to_csv(
        tables_dir / "Source_data_Figure_5_Safety_boundaries.csv",
        index=False,
    )

    fig = plt.figure(figsize=(7.2, 5.6))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.02, 1.18],
        width_ratios=[0.93, 1.18, 1.06],
        hspace=0.58,
        wspace=0.72,
    )

    ax_a = fig.add_subplot(grid[0, :])
    ax_a.plot(
        external_thresholds["Threshold"],
        external_thresholds["FN"],
        color=FIGURE_PALETTE["navy"],
        marker="o",
        markersize=4.2,
        linewidth=1.5,
        label="False negatives",
    )
    ax_a.plot(
        external_thresholds["Threshold"],
        external_thresholds["FP"],
        color=FIGURE_PALETTE["gold"],
        marker="s",
        markersize=4.0,
        linewidth=1.5,
        linestyle="--",
        label="False positives",
    )
    ax_a.axvline(
        threshold,
        color=FIGURE_PALETTE["green"],
        linewidth=1.1,
        linestyle="-.",
    )
    ax_a.annotate(
        f"Locked {threshold:.3f}",
        xy=(threshold, 0.97),
        xycoords=("data", "axes fraction"),
        xytext=(5, -1),
        textcoords="offset points",
        ha="left",
        va="top",
        color=FIGURE_PALETTE["green"],
        fontsize=7,
    )
    ax_a.set_xlabel("Probability threshold")
    ax_a.set_ylabel("Patients")
    ax_a.set_xlim(0.085, 0.515)
    ax_a.set_ylim(bottom=0)
    ax_a.grid(axis="y", color=FIGURE_PALETTE["cream"], linewidth=0.7)
    ax_a.legend(frameon=False, loc="upper right", ncol=2)
    panel_header(ax_a, "A", "Threshold trade-off")

    ax_b = fig.add_subplot(grid[1, 0])
    x_missing = np.arange(len(missingness))
    ax_b.errorbar(
        x_missing,
        missingness["Sensitivity"],
        yerr=np.vstack(
            [
                missingness["Sensitivity"]
                - missingness["Sensitivity_CI_low"],
                missingness["Sensitivity_CI_high"]
                - missingness["Sensitivity"],
            ]
        ),
        color=FIGURE_PALETTE["navy"],
        marker="o",
        markersize=5.0,
        linestyle="none",
        capsize=2.5,
        linewidth=1.0,
    )
    ax_b.axhline(
        0.776,
        color=FIGURE_PALETTE["green"],
        linewidth=1.0,
        linestyle="--",
        label="Overall 0.776",
    )
    for x_value, sensitivity in zip(x_missing, missingness["Sensitivity"]):
        ax_b.text(
            x_value,
            min(0.96, sensitivity + 0.055),
            f"{sensitivity:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax_b.set_xticks(x_missing)
    ax_b.set_xticklabels(
        [
            f"{level}\n(n={int(n):,})"
            for level, n in zip(missingness["Level"], missingness["N"])
        ]
    )
    ax_b.set_ylim(0, 1)
    ax_b.set_xlabel("Missing predictors")
    ax_b.set_ylabel("Sensitivity")
    ax_b.grid(axis="y", color=FIGURE_PALETTE["cream"], linewidth=0.7)
    ax_b.legend(frameon=False, loc="lower left", fontsize=6.3)
    panel_header(ax_b, "B", "Missing-feature burden")

    ax_c = fig.add_subplot(grid[1, 1])
    y_masking = np.arange(len(masking))
    masking_colors = [
        FIGURE_PALETTE["gold"] if rank < 3 else FIGURE_PALETTE["periwinkle"]
        for rank in range(len(masking))
    ]
    ax_c.barh(
        y_masking,
        masking["Delta_sensitivity"],
        color=masking_colors,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_c.set_yticks(y_masking)
    ax_c.set_yticklabels(
        [SHORT_NAMES[name] for name in masking["Feature masked"]],
        fontsize=6.3,
    )
    ax_c.invert_yaxis()
    ax_c.axvline(0, color=FIGURE_PALETTE["green"], linewidth=0.9)
    for y_value, delta in zip(y_masking, masking["Delta_sensitivity"]):
        ax_c.text(
            delta - 0.002,
            y_value,
            f"{delta:.3f}",
            ha="right",
            va="center",
            fontsize=6.1,
        )
    ax_c.set_xlim(-0.115, 0.005)
    ax_c.set_xlabel("Change in sensitivity")
    ax_c.grid(axis="x", color=FIGURE_PALETTE["cream"], linewidth=0.7)
    panel_header(ax_c, "C", "Single-input masking")

    ax_d = fig.add_subplot(grid[1, 2])
    age_wide = age.pivot(
        index="Level",
        columns="Dataset",
        values=["AUROC", "AUROC_CI_low", "AUROC_CI_high"],
    ).reindex(age_order)
    y_age = np.arange(len(age_wide))
    for y_value, (_, values) in zip(y_age, age_wide.iterrows()):
        ax_d.plot(
            [
                values[("AUROC", "Internal test")],
                values[("AUROC", "External validation")],
            ],
            [y_value, y_value],
            color=FIGURE_PALETTE["cream"],
            linewidth=2.0,
            zorder=1,
        )
    ax_d.errorbar(
        age_wide[("AUROC", "Internal test")],
        y_age,
        xerr=np.vstack(
            [
                age_wide[("AUROC", "Internal test")]
                - age_wide[("AUROC_CI_low", "Internal test")],
                age_wide[("AUROC_CI_high", "Internal test")]
                - age_wide[("AUROC", "Internal test")],
            ]
        ),
        color=FIGURE_PALETTE["navy"],
        marker="o",
        markersize=4.8,
        linestyle="none",
        capsize=2.2,
        label="Internal",
        zorder=2,
    )
    ax_d.errorbar(
        age_wide[("AUROC", "External validation")],
        y_age,
        xerr=np.vstack(
            [
                age_wide[("AUROC", "External validation")]
                - age_wide[("AUROC_CI_low", "External validation")],
                age_wide[("AUROC_CI_high", "External validation")]
                - age_wide[("AUROC", "External validation")],
            ]
        ),
        color=FIGURE_PALETTE["gold"],
        marker="s",
        markersize=4.6,
        linestyle="none",
        capsize=2.2,
        label="External",
        zorder=3,
    )
    for y_value, (_, values) in zip(y_age, age_wide.iterrows()):
        ax_d.text(
            values[("AUROC", "Internal test")] - 0.004,
            y_value - 0.13,
            f"{values[('AUROC', 'Internal test')]:.3f}",
            ha="right",
            va="center",
            fontsize=6.0,
            color=FIGURE_PALETTE["navy"],
        )
        ax_d.text(
            values[("AUROC", "External validation")] + 0.004,
            y_value + 0.13,
            f"{values[('AUROC', 'External validation')]:.3f}",
            ha="left",
            va="center",
            fontsize=6.0,
            color=FIGURE_PALETTE["gold"],
        )
    ax_d.set_yticks(y_age)
    ax_d.set_yticklabels(["<65", "65–79", "≥80"])
    ax_d.invert_yaxis()
    ax_d.set_ylim(2.25, -0.25)
    ax_d.set_xlim(0.62, 0.82)
    ax_d.set_xlabel("AUROC")
    ax_d.set_ylabel("Age, years")
    ax_d.grid(axis="x", color=FIGURE_PALETTE["cream"], linewidth=0.7)
    ax_d.text(
        0.815,
        1.34,
        "● Internal",
        color=FIGURE_PALETTE["navy"],
        ha="right",
        va="center",
        fontsize=6.3,
    )
    ax_d.text(
        0.815,
        1.62,
        "■ MIMIC-III",
        color=FIGURE_PALETTE["gold"],
        ha="right",
        va="center",
        fontsize=6.3,
    )
    panel_header(ax_d, "D", "Age subgroups")

    fig.subplots_adjust(left=0.09, right=0.985, top=0.95, bottom=0.09)
    save_figure(fig, figures_dir / "Figure_5_Safety_boundaries", tight=False)
    plt.close(fig)


def baseline_score_table(
    y_internal: pd.Series,
    p_internal: np.ndarray,
    meta_internal: pd.DataFrame,
    y_external: pd.Series,
    p_external: np.ndarray,
    meta_external: pd.DataFrame,
    tables_dir: Path,
) -> None:
    rows = []
    for dataset, y, probabilities, metadata in [
        ("Internal test", y_internal, p_internal, meta_internal),
        ("External validation", y_external, p_external, meta_external),
    ]:
        for score in ["sofa", "sapsii"]:
            mask = metadata[score].notna()
            if mask.sum() < 30 or np.unique(y[mask]).size < 2:
                continue
            values = metadata.loc[mask, score].to_numpy()
            outcomes = y[mask].to_numpy()
            model_probabilities = probabilities[mask.to_numpy()]
            rng = np.random.default_rng(SEED)
            bootstrap_rows = []
            for _ in range(N_BOOTSTRAP):
                indices = rng.integers(0, len(outcomes), len(outcomes))
                sampled_y = outcomes[indices]
                if np.unique(sampled_y).size < 2:
                    continue
                score_auc = roc_auc_score(sampled_y, values[indices])
                model_auc = roc_auc_score(
                    sampled_y,
                    model_probabilities[indices],
                )
                bootstrap_rows.append(
                    {
                        "AUROC": score_auc,
                        "PR_AUC": average_precision_score(
                            sampled_y,
                            values[indices],
                        ),
                        "Delta_AUROC_SepXAI_minus_score": model_auc - score_auc,
                    }
                )
            bootstrap_frame = pd.DataFrame(bootstrap_rows)
            score_auc = roc_auc_score(outcomes, values)
            model_auc = roc_auc_score(outcomes, model_probabilities)
            rows.append(
                {
                    "Dataset": dataset,
                    "Score": score.upper() if score == "sofa" else "SAPS II",
                    "N": int(mask.sum()),
                    "Events": int(y[mask].sum()),
                    "AUROC": score_auc,
                    "AUROC_CI_low": bootstrap_frame["AUROC"].quantile(0.025),
                    "AUROC_CI_high": bootstrap_frame["AUROC"].quantile(0.975),
                    "PR_AUC": average_precision_score(outcomes, values),
                    "PR_AUC_CI_low": bootstrap_frame["PR_AUC"].quantile(0.025),
                    "PR_AUC_CI_high": bootstrap_frame["PR_AUC"].quantile(0.975),
                    "SepXAI_AUROC_same_patients": model_auc,
                    "Delta_AUROC_SepXAI_minus_score": model_auc - score_auc,
                    "Delta_AUROC_CI_low": bootstrap_frame[
                        "Delta_AUROC_SepXAI_minus_score"
                    ].quantile(0.025),
                    "Delta_AUROC_CI_high": bootstrap_frame[
                        "Delta_AUROC_SepXAI_minus_score"
                    ].quantile(0.975),
                }
            )
    pd.DataFrame(rows).to_csv(
        tables_dir / "Table_S12_clinical_score_comparison.csv", index=False
    )


def reporting_audit_table(tables_dir: Path) -> None:
    """Write a concise TRIPOD+AI/PROBAST+AI reporting and limitation audit."""
    rows = [
        {
            "Domain": "Participants and data sources",
            "Evidence in R2": (
                "Database versions, eligibility, first-stay rule, cohort flow, "
                "index time, observation window, and prediction time are reported."
            ),
            "Residual concern": (
                "Both databases are from one center; calendar and patient overlap "
                "cannot be excluded."
            ),
        },
        {
            "Domain": "Predictors",
            "Evidence in R2": (
                "Availability before prediction, units, aggregation, ranges, "
                "missingness, and harmonization are documented."
            ),
            "Residual concern": (
                "Retrospective measurement and database-specific coding can differ."
            ),
        },
        {
            "Domain": "Outcome",
            "Evidence in R2": (
                "Thirty-day all-cause mortality from hospital admission is defined."
            ),
            "Residual concern": "Outcome ascertainment depends on database coding.",
        },
        {
            "Domain": "Model development",
            "Evidence in R2": (
                "Patient-level splitting precedes preprocessing; selection, tuning, "
                "calibration, and threshold choice are separated."
            ),
            "Residual concern": (
                "No formal sample-size calculation was performed; all eligible "
                "records were used."
            ),
        },
        {
            "Domain": "Model evaluation",
            "Evidence in R2": (
                "Locked testing reports discrimination, calibration, clinical "
                "utility, confusion counts, and bootstrap intervals."
            ),
            "Residual concern": (
                "No patient-level prediction interval or prospective impact "
                "evaluation is available."
            ),
        },
        {
            "Domain": "Fairness and robustness",
            "Evidence in R2": (
                "Age, sex, race, SOFA, missingness, missing-input, threshold, and "
                "out-of-distribution analyses are reported."
            ),
            "Residual concern": (
                "Subgroup analyses are exploratory; some strata are small and no "
                "formal interaction testing was performed."
            ),
        },
        {
            "Domain": "Open science",
            "Evidence in R2": (
                "The MIMIC-IV extraction query; MIMIC-III source publication, "
                "supplement filename, analytic-file checksum, import and landmark "
                "logic; software versions; seed; grids; model version; and model "
                "SHA-256 are supplied."
            ),
            "Residual concern": (
                "Patient-level MIMIC data remain subject to credentialed PhysioNet "
                "access and cannot be redistributed."
            ),
        },
        {
            "Domain": "Patient and public involvement",
            "Evidence in R2": (
                "No patient or public involvement occurred in this deidentified "
                "secondary-database study."
            ),
            "Residual concern": (
                "Future workflow and impact studies should include intended users "
                "and patient representatives."
            ),
        },
    ]
    pd.DataFrame(rows).to_csv(
        tables_dir / "Table_S14_reporting_and_risk_of_bias_audit.csv",
        index=False,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analysis_main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    models_dir = output_root / "model"
    for directory in [figures_dir, tables_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    set_style()

    iv, merged_iv, audit_iv = load_mimic_iv(args.data_root)
    iii, raw_iii, audit_iii = load_mimic_iii(args.data_root)
    audit_iv.assign(dataset="MIMIC-IV").to_csv(
        tables_dir / "Table_S1A_data_quality_MIMIC_IV.csv", index=False
    )
    audit_iii.assign(dataset="MIMIC-III").to_csv(
        tables_dir / "Table_S1B_data_quality_MIMIC_III.csv", index=False
    )
    make_baseline_tables(merged_iv, raw_iii, iv, iii, tables_dir)

    development_indices, test_indices = train_test_split(
        np.arange(len(iv.y)),
        test_size=0.20,
        random_state=SEED,
        stratify=iv.y,
    )
    fit_indices, calibration_indices = train_test_split(
        development_indices,
        test_size=0.20,
        random_state=SEED,
        stratify=iv.y.iloc[development_indices],
    )
    split_table = pd.DataFrame(
        [
            {
                "Subset": "Model-training",
                "N": len(fit_indices),
                "Events": int(iv.y.iloc[fit_indices].sum()),
            },
            {
                "Subset": "Calibration/threshold",
                "N": len(calibration_indices),
                "Events": int(iv.y.iloc[calibration_indices].sum()),
            },
            {
                "Subset": "Internal test",
                "N": len(test_indices),
                "Events": int(iv.y.iloc[test_indices].sum()),
            },
            {
                "Subset": "External validation",
                "N": len(iii.y),
                "Events": int(iii.y.sum()),
            },
        ]
    )
    split_table.to_csv(tables_dir / "cohort_split_counts.csv", index=False)
    pd.DataFrame(
        [
            {
                "Dataset": "MIMIC-IV",
                "Rule": "Alive at ICU admission +24 h",
                "Excluded": int(iv.metadata.attrs["landmark_excluded"]),
            },
            {
                "Dataset": "MIMIC-III",
                "Rule": "ICU length of stay >=1.0 day",
                "Excluded": int(iii.metadata.attrs["landmark_excluded"]),
            },
        ]
    ).to_csv(tables_dir / "cohort_landmark_exclusions.csv", index=False)

    X_fit = iv.X.iloc[fit_indices].copy()
    y_fit = iv.y.iloc[fit_indices].copy()
    X_cal = iv.X.iloc[calibration_indices].copy()
    y_cal = iv.y.iloc[calibration_indices].copy()
    X_test = iv.X.iloc[test_indices].copy()
    y_test = iv.y.iloc[test_indices].copy()
    meta_test = iv.metadata.iloc[test_indices].copy()

    selected_features = fit_feature_selection(
        X_fit, y_fit, tables_dir, figures_dir
    )
    # The logistic L1 audit determines the retained harmonized predictors. If a future
    # rerun drops one, all downstream models use exactly the selected schema.
    X_fit = X_fit[selected_features]
    X_cal = X_cal[selected_features]
    X_test = X_test[selected_features]
    X_external = iii.X[selected_features].copy()

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    fitted_searches: dict[str, GridSearchCV] = {}
    comparison_probabilities: dict[str, np.ndarray] = {}
    tuning_rows = []
    for name, (pipeline, grid) in build_model_specs(args.quick).items():
        search = GridSearchCV(
            pipeline,
            grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=3,
            refit=True,
            return_train_score=False,
        )
        search.fit(X_fit, y_fit)
        fitted_searches[name] = search
        probabilities = search.best_estimator_.predict_proba(X_test)[:, 1]
        comparison_probabilities[name] = probabilities
        tuning_rows.append(
            {
                "Model": name,
                "Best_CV_AUROC": search.best_score_,
                "Best_parameters": json.dumps(search.best_params_, sort_keys=True),
                "Grid": json.dumps(grid, sort_keys=True),
            }
        )
    pd.DataFrame(tuning_rows).to_csv(
        tables_dir / "Table_S13_hyperparameter_tuning.csv", index=False
    )

    selected_model_name = max(
        fitted_searches,
        key=lambda model_name: fitted_searches[model_name].best_score_,
    )
    fitted_base = fitted_searches[selected_model_name].best_estimator_
    fitted_base.fit(X_fit, y_fit)
    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(fitted_base),
        method="sigmoid",
    )
    calibrated_model.fit(X_cal, y_cal)
    p_cal = calibrated_model.predict_proba(X_cal)[:, 1]
    operating_threshold = choose_threshold(y_cal, p_cal)

    p_internal = calibrated_model.predict_proba(X_test)[:, 1]
    p_external = calibrated_model.predict_proba(X_external)[:, 1]
    comparison_probabilities["SepXAI"] = p_internal
    metrics = pd.concat(
        [
            summarize_with_ci(
                "Internal test",
                y_test.to_numpy(),
                p_internal,
                operating_threshold,
                250 if args.quick else N_BOOTSTRAP,
            ),
            summarize_with_ci(
                "External validation",
                iii.y.to_numpy(),
                p_external,
                operating_threshold,
                250 if args.quick else N_BOOTSTRAP,
            ),
        ],
        ignore_index=True,
    )
    metrics.to_csv(tables_dir / "Table_2_performance_with_95CI.csv", index=False)

    plot_model_performance(
        comparison_probabilities,
        y_test,
        iii.y,
        p_internal,
        p_external,
        operating_threshold,
        figures_dir,
        tables_dir,
    )
    plot_calibration_and_dca(
        y_test,
        p_internal,
        iii.y,
        p_external,
        operating_threshold,
        figures_dir,
        tables_dir,
    )
    safety_analyses(
        calibrated_model,
        X_fit,
        y_cal,
        p_cal,
        X_test,
        y_test,
        p_internal,
        meta_test,
        X_external,
        iii.y,
        p_external,
        iii.metadata,
        operating_threshold,
        tables_dir,
    )
    plot_safety_boundaries(
        figures_dir,
        tables_dir,
        operating_threshold,
    )
    plot_shap_and_lime(
        fitted_base,
        X_fit,
        X_test,
        y_test,
        p_internal,
        figures_dir,
        tables_dir,
    )
    baseline_score_table(
        y_test,
        p_internal,
        meta_test,
        iii.y,
        p_external,
        iii.metadata,
        tables_dir,
    )
    reporting_audit_table(tables_dir)

    model_path = models_dir / "sepxai_pipeline.joblib"
    joblib.dump(calibrated_model, model_path)
    explanation_reference = {}
    for feature in selected_features:
        numeric = pd.to_numeric(X_fit[feature], errors="coerce")
        quantiles = numeric.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
        explanation_reference[feature] = {
            "q01": float(quantiles.loc[0.01]),
            "q25": float(quantiles.loc[0.25]),
            "median": float(quantiles.loc[0.50]),
            "q75": float(quantiles.loc[0.75]),
            "q99": float(quantiles.loc[0.99]),
        }

    metadata = {
        "model_version": MODEL_VERSION,
        "sha256": sha256(model_path),
        "random_seed": SEED,
        "features": selected_features,
        "display_names": {f: DISPLAY_NAMES[f] for f in selected_features},
        "units": {f: UNITS[f] for f in selected_features},
        "valid_ranges": {f: VALID_RANGES[f] for f in selected_features},
        "explanation_reference": explanation_reference,
        "outcome": "30-day all-cause mortality from hospital admission",
        "index_time": "first ICU admission in the first hospitalization",
        "observation_window": "ICU admission -6 h to ICU admission +24 h",
        "prediction_time": "end of the 24-h observation window",
        "external_validation": {
            "dataset": "MIMIC-III v1.4",
            "source_doi": MIMIC_III_SOURCE_DOI,
            "source_supplement": MIMIC_III_SOURCE_FILE,
            "source_supplement_sha256": MIMIC_III_SOURCE_SHA256,
            "analytic_file_sha256": iii.metadata.attrs["source_sha256"],
            "reported_analytic_file_sha256": MIMIC_III_ANALYTIC_SHA256,
            "predictor_window": "published first-day ICU summary variables",
            "landmark_rule": "icu_los >= 1.0 day",
            "use": "evaluation only",
        },
        "operating_threshold": operating_threshold,
        "threshold_rationale": (
            f"Highest-specificity threshold achieving sensitivity >= "
            f"{TARGET_SENSITIVITY:.2f} in the held-out calibration subset"
        ),
        "selected_algorithm": selected_model_name,
        "selection_rule": "Highest mean AUROC in development-set cross-validation",
        "best_parameters": fitted_searches[selected_model_name].best_params_,
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__,
            "shap": shap.__version__,
            "matplotlib": mpl.__version__,
            "seaborn": sns.__version__,
        },
    }
    (models_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    predictions = pd.DataFrame(
        {
            "dataset": ["Internal test"] * len(y_test)
            + ["External validation"] * len(iii.y),
            "observed_outcome": np.concatenate([y_test.to_numpy(), iii.y.to_numpy()]),
            "predicted_probability": np.concatenate([p_internal, p_external]),
        }
    )
    predictions.to_csv(tables_dir / "Source_data_predictions.csv", index=False)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


APP_TITLE = "SepXAI"
APP_COPY = {
    "subtitle": "Thirty-day mortality risk in ICU sepsis",
    "warning": (
        "For research use only. SepXAI estimates 30-day mortality risk from the "
        "measurements entered below. It has not been validated for treatment "
        "recommendations or bedside decisions, and it does not provide an individual "
        "prediction interval."
    ),
    "input_heading": "Patient measurements",
    "input_note": (
        "Enter the seven measurements used by SepXAI. Editing a value is a what-if "
        "input perturbation: it changes the estimate and explanations, but it is not "
        "a treatment simulation."
    ),
    "explanation_heading": "Explanation for the entered values",
    "explanation_note": (
        "SHAP and LIME show how the entered values contribute to this estimate within "
        "the model. They do not show what would happen after treatment."
    ),
}
PROJECT_URL = "https://github.com/andrelau0622/30_days_prediction"
PALETTE = {
    "ivory": "#EFE8DD",
    "navy": "#345282",
    "gold": "#DCAF75",
    "blue": "#9CABCC",
    "green": "#48614F",
}

DEFAULT_INPUTS = {
    "lactate_min": 1.40,
    "sbp_ni_min": 91.00,
    "resp_rate_min": 12.00,
    "dbp_ni_max": 83.00,
    "heart_rate_mean": 85.00,
    "spo2_min": 93.00,
    "temperature_max": 37.30,
}


@dataclass(frozen=True)
class ShapExplanation:
    base_value: float
    values: np.ndarray
    feature_names: list[str]
    input_values: np.ndarray
    probability: float


@dataclass(frozen=True)
class LimeTerm:
    feature: str
    input_value: float
    weight: float


def get_app_copy() -> dict[str, object]:
    """Return user-visible interface decisions for testing and rendering."""
    return {
        "title": APP_TITLE,
        **APP_COPY,
        "show_model_provenance": False,
        "show_operating_boundaries": False,
    }


def load_artifacts_from_paths(
    model_path: Path | str, metadata_path: Path | str
) -> tuple[object, dict[str, object]]:
    """Load the locked model and its machine-readable interface contract."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "The locked model or metadata file is missing. Run the analysis workflow first."
        )
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def build_input_schema(metadata: dict[str, object]) -> list[dict[str, object]]:
    """Describe exactly the seven active inputs accepted by the locked model."""
    schema = []
    for feature in metadata["features"]:
        schema.append(
            {
                "feature": feature,
                "display_name": metadata["display_names"][feature],
                "unit": metadata["units"][feature],
                "valid_range": tuple(metadata["valid_ranges"][feature]),
            }
        )
    return schema


def _fitted_imputer_medians(model, features: list[str]) -> dict[str, float]:
    """Read feature medians from the fitted pipeline without changing the model."""
    calibrated = model.calibrated_classifiers_[0]
    frozen = calibrated.estimator
    pipeline = getattr(frozen, "estimator", frozen)
    statistics = pipeline.named_steps["imputer"].statistics_
    return {feature: float(value) for feature, value in zip(features, statistics)}


def _reference_statistics(
    model, metadata: dict[str, object]
) -> dict[str, dict[str, float]]:
    """Return stored training quantiles or a deterministic fitted-model fallback."""
    features = list(metadata["features"])
    stored = metadata.get("explanation_reference")
    if stored and all(feature in stored for feature in features):
        return {
            feature: {key: float(value) for key, value in stored[feature].items()}
            for feature in features
        }

    medians = _fitted_imputer_medians(model, features)
    fallback = {}
    for feature in features:
        low, high = map(float, metadata["valid_ranges"][feature])
        median = min(max(medians[feature], low), high)
        scale = max(min((high - low) / 12.0, max(abs(median) * 0.20, 0.5)), 0.1)
        fallback[feature] = {
            "q01": max(low, median - 2.326 * scale),
            "q25": max(low, median - 0.674 * scale),
            "median": median,
            "q75": min(high, median + 0.674 * scale),
            "q99": min(high, median + 2.326 * scale),
        }
    return fallback


def default_input_values(metadata: dict[str, object]) -> dict[str, float]:
    """Return stable defaults clipped to each model input's accepted range."""
    values = {}
    for feature in metadata["features"]:
        low, high = map(float, metadata["valid_ranges"][feature])
        preferred = float(DEFAULT_INPUTS.get(feature, (low + high) / 2.0))
        values[feature] = min(max(preferred, low), high)
    return values


def _ordered_frame(
    metadata: dict[str, object], values: dict[str, float]
) -> pd.DataFrame:
    features = list(metadata["features"])
    return pd.DataFrame(
        [[float(values[feature]) for feature in features]], columns=features
    )


def predict_probability(
    model, metadata: dict[str, object], values: dict[str, float]
) -> float:
    """Predict calibrated 30-day mortality probability for the current inputs."""
    return float(model.predict_proba(_ordered_frame(metadata, values))[0, 1])


def build_reference_distribution(
    model,
    metadata: dict[str, object],
    *,
    sample_size: int = 768,
    random_seed: int = SEED,
) -> pd.DataFrame:
    """Build a deterministic LIME reference sample from training quantiles."""
    features = list(metadata["features"])
    stats = _reference_statistics(model, metadata)
    rng = np.random.default_rng(random_seed)
    columns = {}
    for feature in features:
        feature_stats = stats[feature]
        median = feature_stats["median"]
        q25 = feature_stats["q25"]
        q75 = feature_stats["q75"]
        q01 = feature_stats["q01"]
        q99 = feature_stats["q99"]
        scale = max((q75 - q25) / 1.349, 1e-6)
        values = rng.normal(median, scale, sample_size)
        columns[feature] = np.clip(values, q01, q99)
    reference = pd.DataFrame(columns, columns=features)
    reference.iloc[0] = [stats[feature]["median"] for feature in features]
    return reference


def _positive_probability_function(model, features: list[str]):
    def predict(array) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(array, dtype=float), columns=features)
        return model.predict_proba(frame)[:, 1]

    return predict


def compute_shap_explanation(
    model, metadata: dict[str, object], values: dict[str, float]
) -> ShapExplanation:
    """Explain the calibrated positive-class probability for the current input."""
    features = list(metadata["features"])
    current = _ordered_frame(metadata, values)
    stats = _reference_statistics(model, metadata)
    background = pd.DataFrame(
        [[stats[feature]["median"] for feature in features]], columns=features
    )
    masker = shap.maskers.Independent(background, max_samples=1)
    explainer = shap.Explainer(
        _positive_probability_function(model, features),
        masker,
        algorithm="exact",
        feature_names=features,
    )
    explanation = explainer(current)
    shap_values = np.asarray(explanation.values[0], dtype=float)
    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    probability = predict_probability(model, metadata, values)
    residual = probability - (base_value + float(shap_values.sum()))
    if abs(residual) > 1e-8:
        shap_values[-1] += residual
    return ShapExplanation(
        base_value=base_value,
        values=shap_values,
        feature_names=features,
        input_values=current.iloc[0].to_numpy(dtype=float),
        probability=probability,
    )


def compute_lime_explanation(
    model, metadata: dict[str, object], values: dict[str, float]
) -> list[LimeTerm]:
    """Fit a deterministic seven-feature LIME surrogate for the current input."""
    features = list(metadata["features"])
    reference = build_reference_distribution(model, metadata)
    explainer = LimeTabularExplainer(
        reference.to_numpy(dtype=float),
        feature_names=features,
        class_names=["Survival", "30-day mortality"],
        mode="classification",
        discretize_continuous=False,
        random_state=SEED,
    )

    def predict(array) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(array, dtype=float), columns=features)
        return model.predict_proba(frame)

    current = _ordered_frame(metadata, values).iloc[0].to_numpy(dtype=float)
    explanation = explainer.explain_instance(
        current,
        predict,
        labels=(1,),
        num_features=len(features),
        num_samples=1600,
    )
    weight_map = dict(explanation.as_map()[1])
    return [
        LimeTerm(
            feature=feature,
            input_value=float(current[index]),
            weight=float(weight_map.get(index, 0.0)),
        )
        for index, feature in enumerate(features)
    ]


def render_shap_force_plot(
    explanation: ShapExplanation, metadata: dict[str, object]
) -> plt.Figure:
    """Render a full-width SHAP force plot in the manuscript palette."""
    display_names = [metadata["display_names"][f] for f in explanation.feature_names]
    figure = shap.force_plot(
        explanation.base_value,
        explanation.values,
        features=explanation.input_values,
        feature_names=display_names,
        matplotlib=True,
        show=False,
        figsize=(14.0, 2.25),
        text_rotation=0,
        contribution_threshold=0.0,
        plot_cmap=[PALETTE["navy"], PALETTE["gold"]],
    )
    axis = figure.axes[0]
    shap_red = np.array([1.0, 13 / 255, 87 / 255])
    shap_blue = np.array([30 / 255, 136 / 255, 229 / 255])
    palette_gold = np.array(mpl.colors.to_rgb(PALETTE["gold"]))
    palette_navy = np.array(mpl.colors.to_rgb(PALETTE["navy"]))
    for gradient in axis.images:
        end_color = (
            PALETTE["navy"]
            if float(gradient.get_extent()[1]) > float(gradient.get_extent()[0])
            else PALETTE["gold"]
        )
        gradient.set_cmap(
            mpl.colors.LinearSegmentedColormap.from_list(
                "sepxai_force", [PALETTE["ivory"], end_color]
            )
        )
    for patch in axis.patches:
        face = np.asarray(patch.get_facecolor()[:3])
        edge = np.asarray(patch.get_edgecolor()[:3])
        if np.linalg.norm(face - shap_red) < 0.12:
            patch.set_facecolor(PALETTE["gold"])
        elif np.linalg.norm(face - shap_blue) < 0.12:
            patch.set_facecolor(PALETTE["navy"])
        if np.linalg.norm(edge - np.array([1.0, 195 / 255, 213 / 255])) < 0.2:
            patch.set_edgecolor(PALETTE["gold"])
        elif np.linalg.norm(edge - np.array([209 / 255, 230 / 255, 250 / 255])) < 0.2:
            patch.set_edgecolor(PALETTE["navy"])
    for label in axis.texts:
        color = mpl.colors.to_rgb(label.get_color())
        if np.linalg.norm(np.asarray(color) - shap_red) < 0.12:
            label.set_color(palette_gold)
        elif np.linalg.norm(np.asarray(color) - shap_blue) < 0.12:
            label.set_color(palette_navy)
        elif label.get_color() == "black":
            label.set_color(PALETTE["navy"])
        if " = " in label.get_text():
            label.set_color(PALETTE["navy"])
        if label.get_text().startswith("Maximum temperature ="):
            label.set_y(-0.27)
        if label.get_text() in {
            "higher",
            "lower",
            "$\\leftarrow$",
            "$\\rightarrow$",
            "base value",
            "f(x)",
            f"{explanation.probability:.2f}",
        }:
            label.set_visible(False)
    for line in axis.lines:
        color = mpl.colors.to_rgb(line.get_color())
        if np.linalg.norm(np.asarray(color) - shap_red) < 0.12:
            line.set_color(PALETTE["gold"])
        elif np.linalg.norm(np.asarray(color) - shap_blue) < 0.12:
            line.set_color(PALETTE["navy"])
        else:
            line.set_color(PALETTE["blue"])
    axis.text(
        explanation.base_value,
        0.31,
        f"Reference = {explanation.base_value:.3f}",
        ha="center",
        va="bottom",
        color=PALETTE["navy"],
        fontsize=10,
    )
    axis.text(
        explanation.probability,
        0.31,
        f"Current = {explanation.probability:.3f}",
        ha="center",
        va="bottom",
        color=PALETTE["navy"],
        fontsize=10,
        fontweight="bold",
    )
    figure.patch.set_facecolor("white")
    figure.subplots_adjust(left=0.025, right=0.99, top=0.88, bottom=0.28)
    return figure


def render_lime_plot(
    terms: list[LimeTerm], metadata: dict[str, object]
) -> plt.Figure:
    """Render all seven LIME weights as a signed horizontal bar plot."""
    ordered = sorted(terms, key=lambda term: term.weight)
    labels = [
        f"{metadata['display_names'][term.feature]} = {term.input_value:g}"
        for term in ordered
    ]
    weights = np.asarray([term.weight for term in ordered], dtype=float)
    colors = [PALETTE["navy"] if value < 0 else PALETTE["gold"] for value in weights]
    figure, axis = plt.subplots(figsize=(7.2, 3.7))
    axis.barh(np.arange(len(ordered)), weights, color=colors, height=0.68)
    axis.axvline(0.0, color=PALETTE["green"], linewidth=1.0)
    axis.set_yticks(np.arange(len(ordered)), labels=labels)
    axis.set_xlabel("Local LIME weight for 30-day mortality output")
    axis.grid(axis="x", color="#D7DCE6", linewidth=0.6, alpha=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    figure.tight_layout(pad=0.8)
    return figure


def is_streamlit_runtime() -> bool:
    """Return True only when the file is running inside Streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


def render_streamlit_app() -> None:
    """Render the research-only SepXAI interface."""
    import streamlit as st

    app_dir = Path(__file__).resolve().parent
    results_dir = Path(
        os.environ.get(
            "SEPXAI_RESULTS_DIR",
            app_dir / "analysis_results",
        )
    )
    model_path = results_dir / "model" / "sepxai_pipeline.joblib"
    metadata_path = results_dir / "model" / "model_metadata.json"

    st.set_page_config(page_title="SepXAI", page_icon="S", layout="wide")
    st.markdown(
        f"""
        <style>
        :root {{
            --sepxai-ivory: {PALETTE['ivory']};
            --sepxai-navy: {PALETTE['navy']};
            --sepxai-gold: {PALETTE['gold']};
            --sepxai-blue: {PALETTE['blue']};
            --sepxai-green: {PALETTE['green']};
        }}
        .stApp {{ background: var(--sepxai-ivory); }}
        h1, h2, h3 {{ color: var(--sepxai-navy) !important; }}
        [data-testid="stHeader"] {{ height: 0; background: transparent; }}
        [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none; }}
        .block-container {{ padding-top: 1.8rem; max-width: 1480px; }}
        .sepxai-subtitle {{
            color: var(--sepxai-green);
            font-size: 1.1rem;
            font-weight: 600;
            margin: -0.65rem 0 0.8rem 0;
        }}
        .safety-warning, .threshold-note, .explanation-note {{
            border-radius: 0.45rem;
            padding: 0.85rem 1rem;
            line-height: 1.5;
        }}
        .safety-warning {{
            margin: 0.3rem 0 0.8rem 0;
            background: rgba(220, 175, 117, 0.45);
            border-left: 0.42rem solid var(--sepxai-navy);
            color: #28313f;
        }}
        .threshold-note {{
            margin: 0.2rem 0 0.5rem 0;
            background: rgba(156, 171, 204, 0.48);
            border-left: 0.42rem solid var(--sepxai-green);
            color: #24354f;
        }}
        .explanation-note {{
            margin: 0.15rem 0 0.75rem 0;
            background: rgba(255, 255, 255, 0.48);
            border-left: 0.32rem solid var(--sepxai-blue);
            color: #28313f;
        }}
        div[data-testid="stMetricValue"] {{ color: var(--sepxai-navy); }}
        div[data-testid="stNumberInput"] input {{ background: #FFFFFF; }}
        div[data-testid="stLinkButton"] a {{
            background: var(--sepxai-navy);
            border-color: var(--sepxai-navy);
            color: #FFFFFF;
            white-space: nowrap;
        }}
        div[data-testid="stImage"] img {{ background: #FFFFFF; border-radius: 0.35rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title(APP_TITLE)
    st.markdown(
        f'<div class="sepxai-subtitle">{APP_COPY["subtitle"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="safety-warning">{APP_COPY["warning"]}</div>',
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def load_app_artifacts():
        return load_artifacts_from_paths(model_path, metadata_path)

    model, metadata = load_app_artifacts()
    st.caption(
        f"Input window: {metadata['observation_window']}  ·  Risk estimated at: "
        f"{metadata['prediction_time']}  ·  Index: {metadata['index_time']}"
    )
    st.link_button("View code and project", PROJECT_URL)

    st.subheader(APP_COPY["input_heading"])
    st.write(APP_COPY["input_note"])
    schema = build_input_schema(metadata)
    defaults = default_input_values(metadata)
    values: dict[str, float] = {}
    input_columns = st.columns(2)
    for index, item in enumerate(schema):
        low, high = map(float, item["valid_range"])
        feature = item["feature"]
        with input_columns[index % 2]:
            values[feature] = st.number_input(
                f"{item['display_name']} ({item['unit']})",
                min_value=low,
                max_value=high,
                value=float(defaults[feature]),
                step=0.1,
                format="%.2f",
                key=feature,
                help=f"Accepted range: {low:g} to {high:g} {item['unit']}",
            )
            st.caption(f"Accepted range: {low:g} to {high:g} {item['unit']}")

    probability = predict_probability(model, metadata, values)
    threshold = float(metadata["operating_threshold"])
    metric_column, threshold_column = st.columns(2)
    with metric_column:
        st.metric("Estimated 30-day mortality risk", f"{probability:.1%}")
    with threshold_column:
        if probability >= threshold:
            threshold_copy = (
                f"<strong>Above the study threshold ({threshold:.1%}).</strong> "
                "The threshold was set in the calibration cohort and is not a clinical "
                "alert."
            )
        else:
            threshold_copy = (
                f"<strong>Below the study threshold ({threshold:.1%}).</strong> "
                "A result below the threshold does not rule out "
                "deterioration or death."
            )
        st.markdown(
            f'<div class="threshold-note">{threshold_copy}</div>',
            unsafe_allow_html=True,
        )

    st.subheader(APP_COPY["explanation_heading"])
    st.markdown(
        f'<div class="explanation-note">{APP_COPY["explanation_note"]}</div>',
        unsafe_allow_html=True,
    )
    try:
        with st.spinner("Updating the explanation for the current measurements"):
            shap_result = compute_shap_explanation(model, metadata, values)
            lime_result = compute_lime_explanation(model, metadata, values)
            shap_figure = render_shap_force_plot(shap_result, metadata)
            lime_figure = render_lime_plot(lime_result, metadata)
        st.markdown("#### SHAP feature contributions")
        st.pyplot(shap_figure, width="stretch")
        st.caption(
            "Features shift the estimate from the reference value to the value shown "
            "above."
        )
        st.markdown("#### LIME local weights")
        st.pyplot(lime_figure, width="stretch")
        st.caption(
            "Local surrogate weights are recalculated for the current measurements."
        )
        plt.close(shap_figure)
        plt.close(lime_figure)
    except Exception as error:
        st.warning(
            "The probability was calculated, but the explanation panels could not be "
            f"updated: {error}"
        )


if __name__ == "__main__":
    if is_streamlit_runtime():
        render_streamlit_app()
    else:
        analysis_main()
