from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

RAW_DIR = Path(__file__).resolve().parent / "raw"
RAW_CSV = RAW_DIR / "diabetic_data.csv"
VARS_CSV = RAW_DIR / "variables.csv"


"""
Checks if the data is already downloaded, if not then downloads the dataset at:
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Data can be found at ./data/raw/*.csv

diabetic_data - The dataset itself. It has one row per hospital encounter, with all the
feature columns plus the target column readmitted joined on as the last column.
encounter_id and patient_nbr (the UCI "ID" columns) are kept as the first two columns.

variable -  data dictionary from UCI. It has one row per column name, with its 
role (feature or target), type, a short description, units, and whether it has missing values.
Used to look up what a column means and descriptions
"""
def fetch_raw(force=False):
    if RAW_CSV.exists() and not force:
        # '?' is missing; the lab-result label 'None' means no test was taken.
        df = pd.read_csv(RAW_CSV, keep_default_na=False, na_values=["?", ""],
                         low_memory=False)
        if {"encounter_id", "patient_nbr"}.issubset(df.columns):
            for column in ["max_glu_serum", "A1Cresult"]:
                df[column] = df[column].fillna("None")
            return df
        print("Cached data has no patient IDs. Downloading the complete dataset.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ds = fetch_ucirepo(id=296)
    df = pd.concat([ds.data.ids, ds.data.features, ds.data.targets], axis=1)
    # ucimlrepo parses the literal lab category 'None' as NaN; restore it.
    for column in ["max_glu_serum", "A1Cresult"]:
        df[column] = df[column].fillna("None")
    df = df.replace("?", float("nan"))
    df.to_csv(RAW_CSV, index=False)
    ds.variables.to_csv(VARS_CSV, index=False)
    return df


if __name__ == "__main__":
    df = fetch_raw()
    print(df.shape)
