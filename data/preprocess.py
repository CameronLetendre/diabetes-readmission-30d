"""This file prepares our hospital records for models by:

1. Removing stays that ended in death or hospice care.
2. Removing weight because almost all its values are missing.
3. abeling missing information clearly, while keeping “test not performed” as a separate value.
4, Grouping the three diagnosis codes into nine clinical categories plus Missing, as in Strack et al. (2014)
5. Marking whether each stay was followed by a readmission in under 30 days.
6. Keeping patient IDs separate so we can later keep the same patient out of both training and testing.

Run from the project root: python -m data.preprocess.

"""

from pathlib import Path
import re

import pandas as pd

from data.data import fetch_raw


PROCESSED_DIR = Path(__file__).resolve().parent / "processed"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

ID_COLUMNS = ["encounter_id", "patient_nbr"]
NUMERIC_COLUMNS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
DIAGNOSIS_COLUMNS = ["diag_1", "diag_2", "diag_3"]

# Source: IDS_mapping.csv in the official UCI dataset download:
# https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
DEATH_CODES = [11, 19, 20, 21]
HOSPICE_CODES = [13, 14]
MISSING_ID_CODES = {
    "admission_type_id": [5, 6, 8],
    "admission_source_id": [9, 15, 17, 20, 21],
    "discharge_disposition_id": [18, 25, 26],
}


def group_diagnosis(code):
    """Use the nine clinical groups in Strack et al. (2014), plus Missing.

    Source, Table 2: https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/
    Apply the same rules separately to each of the three diagnosis columns.
    """
    if pd.isna(code) or str(code).strip() in ["", "?"]:
        return "Missing"

    code = str(code).strip().upper()
    if re.fullmatch(r"[EV]\d{2,3}(?:\.\d+)?", code):
        return "Other"
    if not re.fullmatch(r"\d{1,3}(?:\.\d+)?", code):
        raise ValueError(f"Unrecognized ICD-9 diagnosis code: {code}")

    value = float(code)
    if not 1 <= value < 1000:
        raise ValueError(f"ICD-9 diagnosis code outside expected range: {code}")
    if 250 <= value < 251:
        return "Diabetes"
    if 390 <= value < 460 or 785 <= value < 786:
        return "Circulatory"
    if 460 <= value < 520 or 786 <= value < 787:
        return "Respiratory"
    if 520 <= value < 580 or 787 <= value < 788:
        return "Digestive"
    if 800 <= value < 1000:
        return "Injury"
    if 710 <= value < 740:
        return "Musculoskeletal"
    if 580 <= value < 630 or 788 <= value < 789:
        return "Genitourinary"
    if 140 <= value < 240:
        return "Neoplasms"
    return "Other"


def preprocess_data(raw):
    """Return prepared encounters and a table recording each exclusion.

    Keep IDs for patient-based splitting and the original readmitted label for
    auditing. Neither IDs nor either target column belongs in model features.
    The command below saves them separately from features.csv.
    """
    required = ID_COLUMNS + NUMERIC_COLUMNS + DIAGNOSIS_COLUMNS + [
        "readmitted", "weight", "race", "payer_code", "medical_specialty",
        "max_glu_serum", "A1Cresult",
    ] + list(MISSING_ID_CODES)
    missing_columns = sorted(set(required) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Required raw columns are missing: {missing_columns}")

    df = raw.copy().replace(["?", ""], float("nan"))
    if df[ID_COLUMNS].isna().any().any():
        raise ValueError("Patient and encounter IDs must not be missing.")
    if df["encounter_id"].duplicated().any():
        raise ValueError("Duplicate encounter IDs found; check the raw data.")
    if not df["readmitted"].isin(["<30", ">30", "NO"]).all():
        raise ValueError("Unexpected or missing readmitted labels.")

    # 1. Define the target before exclusions so every stage reports its balance.
    df["readmit_30"] = df["readmitted"].eq("<30").astype(int)
    for column in MISSING_ID_CODES:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if (df[column].dropna() % 1 != 0).any():
            raise ValueError(f"Expected integer category codes in {column}.")

    counts = [{
        "stage": "raw", "encounters": len(df),
        "patients": df["patient_nbr"].nunique(), "removed_encounters": 0,
        "readmitted_30": int(df["readmit_30"].sum()),
    }]

    # 2. Exclude death and hospice discharges; keep all other encounters.
    for stage, codes in [("exclude_death", DEATH_CODES),
                         ("exclude_hospice", HOSPICE_CODES)]:
        before = len(df)
        df = df.loc[~df["discharge_disposition_id"].isin(codes)].copy()
        counts.append({
            "stage": stage, "encounters": len(df),
            "patients": df["patient_nbr"].nunique(),
            "removed_encounters": before - len(df),
            "readmitted_30": int(df["readmit_30"].sum()),
        })
    if df.empty:
        raise ValueError("No eligible encounters remain after exclusions.")

    # 3. Drop the very sparse weight column. No other feature is dropped here.
    df = df.drop(columns="weight")

    # 4. Distinguish true missing categories from a test that was not performed.
    # 'Not measured' also survives CSV reading, unlike pandas' NA token 'None'.
    for column in ["max_glu_serum", "A1Cresult"]:
        df[column] = df[column].fillna("Not measured").replace("None", "Not measured")

    # Category IDs describe groups, not quantities. Prefix the codes so they
    # remain text categories even when features.csv is read back with pandas.
    for column, missing_codes in MISSING_ID_CODES.items():
        codes = df[column]
        labels = column.removesuffix("_id") + "_" + codes.astype("Int64").astype("string")
        df[column] = labels.mask(codes.isin(missing_codes), "Missing").fillna("Missing")

    excluded_columns = ID_COLUMNS + NUMERIC_COLUMNS + DIAGNOSIS_COLUMNS + [
        "readmitted", "readmit_30"
    ]
    categorical_columns = [column for column in df.columns
                           if column not in excluded_columns]
    for column in categorical_columns:
        df[column] = df[column].fillna("Missing")

    # 5. Replace high-cardinality diagnosis codes with the clinical groups.
    for column in DIAGNOSIS_COLUMNS:
        df[f"{column}_group"] = df[column].map(group_diagnosis)
    df = df.drop(columns=DIAGNOSIS_COLUMNS)

    # Retain repeated patients. Their rows must stay together in the later split.
    return df.reset_index(drop=True), pd.DataFrame(counts)


if __name__ == "__main__":
    df, counts = preprocess_data(fetch_raw())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # All three files have matching row order. Keep IDs and labels out of X.
    features = df.drop(columns=ID_COLUMNS + ["readmitted", "readmit_30"])
    features.to_csv(PROCESSED_DIR / "features.csv", index=False)
    df[["readmit_30"]].to_csv(PROCESSED_DIR / "targets.csv", index=False)
    df[ID_COLUMNS + ["readmitted"]].to_csv(PROCESSED_DIR / "encounters.csv", index=False)
    counts.to_csv(REPORT_DIR / "preprocessing_counts.csv", index=False)

    group_tables = []
    for column in [f"{c}_group" for c in DIAGNOSIS_COLUMNS]:
        table = df[column].value_counts().rename_axis("group").reset_index(name="encounters")
        table.insert(0, "column", column)
        group_tables.append(table)
    pd.concat(group_tables, ignore_index=True).to_csv(
        REPORT_DIR / "diagnosis_group_counts.csv", index=False
    )

    print(counts.to_string(index=False))
    print(f"\nPrepared {len(df):,} encounters from {df['patient_nbr'].nunique():,} patients.")
    print(f"Features: {features.shape[1]}; 30-day readmission rate: {df['readmit_30'].mean():.2%}")
    print(f"Data saved to {PROCESSED_DIR}")
    print(f"Exclusion and diagnosis summaries saved to {REPORT_DIR}")
