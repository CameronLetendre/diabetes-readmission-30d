"""This file prepares our hospital records for models by:

1. Marking whether each stay was followed by a readmission in under 30 days.
2. Removing stays that ended in death or hospice care.
3. Removing the few stays whose gender is recorded as Unknown/Invalid.
4. Removing weight because almost all its values are missing.
5. Grouping the three ID code columns by clinical meaning.
6. Labeling missing information clearly, while keeping "test not performed" as a separate value.
7. Making age a number, flagging lab tests as ordered or not, and simplifying payer and specialty.
8. Grouping the three diagnosis codes into nine clinical categories plus Missing, as in Strack et al. (2014).
9. Keeping patient IDs separate so we can later keep the same patient out of both training and testing.

Run from the project root: python -m scripts.preprocess.

"""

from pathlib import Path
import re
import sys
import pandas as pd

# Let "python scripts/preprocess.py" (or the IDE Run button) find the scripts package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.data import fetch_raw
from scripts.id_groups import apply_id_groups

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
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
CONSTANT_DRUGS = [
    "examide", "citoglipton", "acetohexamide", "tolbutamide", "troglitazone",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]
DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "tolazamide", "insulin", "glyburide-metformin",
]
AGE_ORDER = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
             "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]


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

    # 3. Exclude the few encounters whose gender is Unknown/Invalid: a data entry error.
    before = len(df)
    df = df.loc[df["gender"] != "Unknown/Invalid"].copy()
    counts.append({
        "stage": "exclude_invalid_gender", "encounters": len(df),
        "patients": df["patient_nbr"].nunique(),
        "removed_encounters": before - len(df),
        "readmitted_30": int(df["readmit_30"].sum()),
    })

    # 4. Drop the very sparse weight column. No other feature is dropped here.
    df = df.drop(columns=["weight"] + CONSTANT_DRUGS)

    # 5. Group the ID codes by meaning. Admission type, discharge disposition, and
    # admission source are stored as integers, but they are labels. Several codes
    # mean NULL or Not Mapped, and most are rare, so each code is mapped to a larger
    # group that shares a clinical meaning. The raw integer columns are dropped.
    df = apply_id_groups(df)

    # 6. Distinguish true missing categories from a test that was not performed.
    # 'Not measured' also survives CSV reading, unlike pandas' NA token 'None'.
    for column in ["max_glu_serum", "A1Cresult"]:
        df[column] = df[column].fillna("Not measured").replace("None", "Not measured")

    excluded_columns = ID_COLUMNS + NUMERIC_COLUMNS + DIAGNOSIS_COLUMNS + [
        "readmitted", "readmit_30"
    ]
    categorical_columns = [column for column in df.columns
                           if column not in excluded_columns]
    for column in categorical_columns:
        df[column] = df[column].fillna("Missing")

    # 7. Make age a number. The buckets have a real order, so map them 0 through 9.
    df["age"] = df["age"].map({label: i for i, label in enumerate(AGE_ORDER)})
    if df["age"].isna().any():
        raise ValueError("Unexpected age bucket in the raw data.")

    # 8. Flag lab tests as tested or not. A blank result means the test was never
    # ordered, and Strack et al. (2014) found that ordering the test matters.
    df["a1c_tested"] = df["A1Cresult"].ne("Not measured").astype(int)
    df["glucose_tested"] = df["max_glu_serum"].ne("Not measured").astype(int)
    df["a1c_tested_and_changed"] = df["a1c_tested"] * df["change"].eq("Ch").astype(int)

    # 9. Simplify payer and specialty. The 10 most common specialties cover 91.8% of
    # rows; the rest fold into Other. Medicare and Medicaid flags let us check later
    # whether the model works equally well across payers.
    top = df["medical_specialty"].value_counts().index[:10]
    df["medical_specialty"] = df["medical_specialty"].where(
        df["medical_specialty"].isin(top), "Other")

    df["medicare"] = df["payer_code"].eq("MC").astype(int)
    df["medicaid"] = df["payer_code"].eq("MD").astype(int)

    # 10. Replace high-cardinality diagnosis codes with the clinical groups.
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
