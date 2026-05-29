# NASA C-MAPSS — Remaining Useful Life Prediction

End-to-end deep learning pipelines for predicting Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset.

Each folder is a self-contained notebook for one sub-dataset.

---

## Results

| Dataset | Conditions | Test RMSE | Status |
|---------|------------|-----------|--------|
| FD001   | 1 op condition, 1 fault mode | 14.77 cycles | ✅ Done |
| FD002   | 6 op conditions, 1 fault mode | — | 🔜 Upcoming |
| FD003   | 1 op condition, 2 fault modes | — | 🔜 Upcoming |
| FD004   | 6 op conditions, 2 fault modes | — | 🔜 Upcoming |

### FD001 performance snapshot

| Model | Val RMSE | Val MAE | Test RMSE | Test MAE |
|------|----------|---------|-----------|---------|
| Main LSTM | 12.02 | 9.00 | 14.77 | 11.09 |
| RF (agg features) | 13.05 | 9.97 | 13.45 | 9.81 |
| Stacked RF | 13.45 | 9.63 | 14.72 | 11.03 |

---

## What this project covers

- Loading and parsing raw NASA C-MAPSS txt files
- Computing RUL labels with capping (standard 125-cycle cap)
- Dropping zero-variance sensors
- Train/val split by engine ID (no data leakage)
- Sklearn Pipeline — SimpleImputer + StandardScaler, fit on train only
- Sliding window sequence construction → shape `(N, W, F)`
- Two-layer LSTM with Dropout
- Evaluation with RMSE and MAE on held-out test engines

## Experiments in this repo

This repository contains a small experiment comparing three approaches on FD001:

- A Random Forest baseline trained on aggregated window features (mean, std, min, max, last, slope per sensor).
- A two-layer LSTM that predicts RUL from raw windows.
- A stacked model: the LSTM's prediction is added as a feature to the Random Forest (used as a quick test of stacking).

Latest FD001 run:

| Model | Val RMSE | Val MAE | Test RMSE | Test MAE |
|------|----------|---------|-----------|---------|
| Main LSTM | 12.02 | 9.00 | 14.77 | 11.09 |
| RF (agg features) | 13.05 | 9.97 | 13.45 | 9.81 |
| Stacked RF | 13.45 | 9.63 | 14.72 | 11.03 |

These experiments were performed for testing and model-prototyping only; see `FD001/fd001.ipynb` for the evaluation cells and plots.

---

## Dataset

Download from the NASA Prognostics Data Repository:
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

Place files inside the relevant subfolder:

```
FD001/
  train_FD001.txt
  test_FD001.txt
  RUL_FD001.txt
```

Do not commit the data files — they are in `.gitignore`.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/nasa-cmapss-rul.git
cd nasa-cmapss-rul

uv venv .venv --python "C:\Program Files\Python312\python.exe"

# Windows PowerShell
.venv\Scripts\Activate.ps1

uv pip install --python .venv\Scripts\python.exe -r requirements.txt

jupyter notebook
```

Then open the notebook inside the subfolder you want.

---

## Stack

- Python 3.12
- NumPy, pandas
- scikit-learn
- TensorFlow / Keras
- Matplotlib
- Jupyter

---

## Structure

```
nasa-cmapss-rul/
├── FD001/
│   ├── fd001_rul_prediction.ipynb
│   └── README.md
├── FD002/
├── FD003/
├── FD004/
├── requirements.txt
├── .gitignore
└── README.md
```
