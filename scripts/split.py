from sklearn.model_selection import StratifiedGroupKFold

SEED = 42


# 80/20 train/test split, same patient never on both sides
def patient_split(y, patient_ids, seed=SEED):
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(folds.split(y, y, groups=patient_ids))
    assert set(patient_ids.iloc[train_idx]).isdisjoint(patient_ids.iloc[test_idx])
    return train_idx, test_idx
