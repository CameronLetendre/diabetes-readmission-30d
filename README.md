# Predicting Hospital Readmission Within 30 Days

This project uses hospital records to predict whether a patient with diabetes
will return to the hospital in less than 30 days. It explores the data, prepares
it for analysis, and compares three models: logistic regression, random forest,
and XGBoost. Records from the same patient are kept together when splitting the
data for training and testing.

The data comes from the UCI
[Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
dataset.

## Setup

You need Python installed. Open a terminal in the main project folder and create
an environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Or on macOS / Linux:

```bash
source .venv/bin/activate
```

Install the required packages and Jupyter Notebook:

```bash
python -m pip install -r requirements.txt notebook
```

## Run the project

Download the data from the main project folder. You need internet access for the
first download. Later runs reuse the files in `data/raw/`.

```bash
python scripts/data.py
```

Start Jupyter from the `notebooks/` folder so the notebooks can find their files:

```bash
cd notebooks
python -m notebook
```

Open each notebook in this order and run all its cells from top to bottom:

1. `01_eda.ipynb` — explore the data and make charts.
2. `02_preprocessing.ipynb` — clean and prepare the data.
3. `03_baseline_logreg.ipynb` — train and evaluate logistic regression.
4. `04_tree_models.ipynb` — train and evaluate random forest and XGBoost.
5. `05_comparison.ipynb` — compare the models and review the results.
6. `06_imbalance_threshold.ipynb` — test resampling (including SMOTE) against
   class weights and choose a decision threshold.

Run notebooks 03 and 04 before 05 and 06 because they load the models that 03
and 04 save. Notebook 04 may take a while because it tries several model
settings.

Keep the `scripts/` folder with the notebooks: they use its shared code. Saved
models go in `models/`, charts and result tables go in `reports/figures/`, and
the written report is in `reports/`.
