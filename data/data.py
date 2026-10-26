from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

RAW_DIR = Path("data/raw")
RAW_CSV = RAW_DIR / "diabetic_data.csv"
VARS_CSV = RAW_DIR / "variables.csv"


"""
Checks if the data is already downloaded, if not then downloads the dataset at:
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Data can be found at ./data/raw/*.csv

diabetic_data - The dataset itself. It has one row per hospital encounter, with all the
feature columns plus the target column readmitted joined on as the last column.

variable -  data dictionary from UCI. It has one row per column name, with its 
role (feature or target), type, a short description, units, and whether it has missing values.
Used to look up what a column means and descriptions
"""
def fetch_raw(force=False):
    if RAW_CSV.exists() and not force:
        return pd.read_csv(RAW_CSV)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ds = fetch_ucirepo(id=296)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.to_csv(RAW_CSV, index=False)
    ds.variables.to_csv(VARS_CSV, index=False)
    return df


if __name__ == "__main__":
    df = fetch_raw()
    print(df.shape)
