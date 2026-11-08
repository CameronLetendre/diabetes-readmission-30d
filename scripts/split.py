from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
ROOT = Path(__file__).resolve().parents[1]


# 80/20 train/test split, same patient never on both sides
def patient_split(y, patient_ids, seed=SEED):
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(folds.split(y, y, groups=patient_ids))
    assert set(patient_ids.iloc[train_idx]).isdisjoint(patient_ids.iloc[test_idx])
    return train_idx, test_idx


# Run from the project root: python -m scripts.split
if __name__ == "__main__":
    from scripts.data import fetch_raw
    from scripts.preprocess import preprocess_data

    df, _ = preprocess_data(fetch_raw())
    train_idx, test_idx = patient_split(df["readmit_30"], df["patient_nbr"])

    df["split"] = "train"
    df.loc[df.index[test_idx], "split"] = "test"

    # which encounter is in train or test
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    df[["encounter_id", "patient_nbr", "split"]].to_csv(
        ROOT / "data" / "processed" / "split.csv", index=False)

    # size table
    sizes = df.groupby("split").agg(
        encounters=("encounter_id", "size"),
        patients=("patient_nbr", "nunique"),
        readmit_30_rate=("readmit_30", "mean"),
    ).round(3)
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    sizes.to_csv(ROOT / "reports" / "figures" / "split_sizes.csv")
    print(sizes)