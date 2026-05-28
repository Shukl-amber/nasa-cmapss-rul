# FD001 — RUL Prediction

Single operating condition, single fault mode. The simplest sub-dataset in C-MAPSS and the standard starting point.

---

## Results

| Metric | Val | Test |
|--------|-----|------|
| RMSE   | 14.20 cycles | 14.85 cycles |
| MAE    | 10.15 cycles | 11.09 cycles |

Test RMSE of **14.85 cycles** on 100 unseen engines.

---

## Pipeline

```
train_FD001.txt
  → assign headers (26 columns)
  → compute RUL per engine
  → cap RUL at 125 cycles
  → drop zero-variance sensors (s1, s5, s6, s10, s16, s18, s19 + os1-3)
  → split engines: 80 train / 20 val (by engine ID)
  → fit sklearn Pipeline on train (SimpleImputer → StandardScaler)
  → transform train and val
  → sliding windows W=30 → X shape (N, 30, 14)
  → two-layer LSTM with Dropout
  → evaluate on test_FD001.txt + RUL_FD001.txt
```

---

## Model

```
Input       (None, 30, 14)
LSTM        (None, 30, 64)   return_sequences=True
Dropout     0.2
LSTM        (None, 32)
Dropout     0.2
Dense       (None, 32)       relu
Dense       (None, 1)        linear
```

- Optimizer: Adam
- Loss: MSE
- Early stopping: patience=5, restore best weights

---

## Training curve

Both train and val loss converge cleanly. No overfitting after RUL capping.

---

## Files needed

Place these in this folder before running the notebook:

```
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

Download from:
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

---

## Key decisions

**Why cap RUL at 125?**
Early in engine life there is no degradation signal. High RUL labels (200–300) are noisy and hurt training. Capping treats all healthy states equally and focuses the model on the degradation phase.

**Why split by engine ID?**
Splitting by row would mix early and late cycles from the same engine into train and val. The model would leak future knowledge. Engine-level split is the only honest evaluation.

**Why fit scaler on train only?**
Fitting on the full dataset lets val/test statistics influence the scaler. This is data leakage. The scaler must only see train data.
