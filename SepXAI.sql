-- SepXAI MIMIC-IV extraction query
-- Dataset version: MIMIC-IV v3.0
-- Index time: first ICU admission during the first hospitalization
-- Observation window: ICU admission -6 hours through ICU admission +24 hours
-- Prediction time: end of the observation window
-- Outcome: death within 30 days of hospital admission
--
-- IMPORTANT
-- 1. All predictor event times are constrained to the observation window.
-- 2. Sepsis-3 eligibility is required on or before the prediction time.
-- 3. Only one first-hospital/first-ICU record is retained per patient.
-- 4. Patients must be alive at the +24-hour prediction landmark.
-- 5. Physiologically impossible values are converted to NULL before aggregation.
-- 6. Post-window treatments, intervention durations, and lengths of stay are not
--    candidate predictors.
--
-- The project/schema prefix may need to be adapted to the local MIMIC deployment.

WITH eligible_cohort AS (
    SELECT
        detail.subject_id,
        detail.hadm_id,
        detail.stay_id,
        detail.admittime,
        detail.icu_intime,
        DATETIME_ADD(detail.icu_intime, INTERVAL '24' HOUR) AS prediction_time,
        s3.suspected_infection_time,
        s3.sofa_time,
        s3.sofa_score
    FROM mimiciv_derived.icustay_detail AS detail
    INNER JOIN mimiciv_derived.sepsis3 AS s3
        ON detail.subject_id = s3.subject_id
       AND detail.stay_id = s3.stay_id
    INNER JOIN mimiciv_hosp.admissions AS admission
        ON detail.subject_id = admission.subject_id
       AND detail.hadm_id = admission.hadm_id
    WHERE s3.sepsis3 = 't'
      AND detail.first_hosp_stay = 't'
      AND detail.first_icu_stay = 't'
      AND detail.admission_age >= 18
      AND detail.admission_age <= 100
      AND s3.sofa_time <= DATETIME_ADD(detail.icu_intime, INTERVAL '24' HOUR)
      AND (
          admission.deathtime IS NULL
          OR admission.deathtime >=
              DATETIME_ADD(detail.icu_intime, INTERVAL '24' HOUR)
      )
),

vital_aggregates AS (
    SELECT
        cohort.subject_id,
        cohort.hadm_id,
        cohort.stay_id,
        MIN(
            CASE
                WHEN vital.sbp_ni BETWEEN 30 AND 300 THEN vital.sbp_ni
                ELSE NULL
            END
        ) AS sbp_ni_min,
        MAX(
            CASE
                WHEN vital.spo2 BETWEEN 30 AND 100 THEN vital.spo2
                ELSE NULL
            END
        ) AS spo2_max,
        MIN(
            CASE
                WHEN vital.resp_rate BETWEEN 1 AND 80 THEN vital.resp_rate
                ELSE NULL
            END
        ) AS resp_rate_min,
        MAX(
            CASE
                WHEN vital.dbp_ni BETWEEN 10 AND 200 THEN vital.dbp_ni
                ELSE NULL
            END
        ) AS dbp_ni_max,
        AVG(
            CASE
                WHEN vital.heart_rate BETWEEN 20 AND 250 THEN vital.heart_rate
                ELSE NULL
            END
        ) AS heart_rate_mean,
        MIN(
            CASE
                WHEN vital.spo2 BETWEEN 30 AND 100 THEN vital.spo2
                ELSE NULL
            END
        ) AS spo2_min,
        MAX(
            CASE
                WHEN vital.temperature BETWEEN 25 AND 45 THEN vital.temperature
                ELSE NULL
            END
        ) AS temperature_max
    FROM eligible_cohort AS cohort
    LEFT JOIN mimiciv_derived.vitalsign AS vital
        ON cohort.subject_id = vital.subject_id
       AND cohort.stay_id = vital.stay_id
       AND vital.charttime BETWEEN
           DATETIME_SUB(cohort.icu_intime, INTERVAL '6' HOUR)
           AND cohort.prediction_time
    GROUP BY cohort.subject_id, cohort.hadm_id, cohort.stay_id
),

lactate_aggregate AS (
    SELECT
        cohort.subject_id,
        cohort.hadm_id,
        cohort.stay_id,
        MIN(
            CASE
                WHEN bg.lactate BETWEEN 0.2 AND 30 THEN bg.lactate
                ELSE NULL
            END
        ) AS lactate_min
    FROM eligible_cohort AS cohort
    LEFT JOIN mimiciv_derived.bg AS bg
        ON cohort.subject_id = bg.subject_id
       AND cohort.hadm_id = bg.hadm_id
       AND bg.charttime BETWEEN
           DATETIME_SUB(cohort.icu_intime, INTERVAL '6' HOUR)
           AND cohort.prediction_time
    GROUP BY cohort.subject_id, cohort.hadm_id, cohort.stay_id
),

outcomes AS (
    SELECT
        cohort.subject_id,
        cohort.hadm_id,
        cohort.stay_id,
        CASE
            WHEN admission.deathtime IS NOT NULL
             AND admission.deathtime <=
                 DATETIME_ADD(admission.admittime, INTERVAL '30' DAY)
            THEN 1
            ELSE 0
        END AS death_within_hosp_30days
    FROM eligible_cohort AS cohort
    INNER JOIN mimiciv_hosp.admissions AS admission
        ON cohort.subject_id = admission.subject_id
       AND cohort.hadm_id = admission.hadm_id
)

SELECT
    cohort.subject_id,
    cohort.hadm_id,
    cohort.stay_id,
    cohort.admittime,
    cohort.icu_intime,
    cohort.prediction_time,
    cohort.suspected_infection_time,
    cohort.sofa_time,
    cohort.sofa_score,
    lactate.lactate_min,
    vital.sbp_ni_min,
    vital.spo2_max,
    vital.resp_rate_min,
    vital.dbp_ni_max,
    vital.heart_rate_mean,
    vital.spo2_min,
    vital.temperature_max,
    outcome.death_within_hosp_30days
FROM eligible_cohort AS cohort
LEFT JOIN vital_aggregates AS vital
    ON cohort.subject_id = vital.subject_id
   AND cohort.hadm_id = vital.hadm_id
   AND cohort.stay_id = vital.stay_id
LEFT JOIN lactate_aggregate AS lactate
    ON cohort.subject_id = lactate.subject_id
   AND cohort.hadm_id = lactate.hadm_id
   AND cohort.stay_id = lactate.stay_id
INNER JOIN outcomes AS outcome
    ON cohort.subject_id = outcome.subject_id
   AND cohort.hadm_id = outcome.hadm_id
   AND cohort.stay_id = outcome.stay_id;
