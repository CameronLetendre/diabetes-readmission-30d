# diabetes-readmission-30d

Predicting 30-day hospital readmission for diabetic patients using the
[Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
dataset from the UCI Machine Learning Repository.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Loading the data

The raw data is not committed to the repo (it is about 19 MB and is ignored by
`.gitignore`). Fetch it once by running the loader from the project root:

```bash
python src/data.py
```

The first run downloads the dataset from UCI via the `ucimlrepo` package and
saves two files under `data/raw/`:

| File | Contents |
| --- | --- |
| `diabetic_data.csv` | The dataset. One row per hospital encounter (~101,766 rows), all feature columns plus the target column `readmitted` as the last column. |
| `variables.csv` | The UCI data dictionary. One row per column with its role, type, description, units, and whether it has missing values. |

Later runs find `diabetic_data.csv` on disk and skip the download, so
downstream scripts never need network access.

### Using it from other code

```python
from src.data import fetch_raw

df = fetch_raw()            # reads data/raw/diabetic_data.csv, downloading it first if missing
df = fetch_raw(force=True)  # re-download even if the file already exists
```

`fetch_raw` returns a pandas DataFrame. Paths are relative to the project root,
so run scripts from there.

## Project layout

```
data/raw/        raw CSVs fetched by src/data.py (git-ignored)
src/data.py      dataset loader
requirements.txt pinned Python dependencies
```
