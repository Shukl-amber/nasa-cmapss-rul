# FD001 — RUL Prediction

Single operating condition, single fault mode. This notebook predicts remaining useful life for the NASA C-MAPSS FD001 subset.

## Summary

The notebook:

- loads `train_FD001.txt`
- checks the raw data
- prints sensor variance in ascending order
- computes RUL per engine and caps it at 125 cycles
- drops `os1`, `os2`, `os3`, `s1`, `s5`, `s6`, `s10`, `s16`, `s18`, `s19`
- splits engines 80/20 for train and validation
- fits `SimpleImputer` + `StandardScaler` on train only
- builds windows with `W = 40`
- trains the main two-layer LSTM
- evaluates on validation and test sets
- compares against a Random Forest baseline and a stacked RF + LSTM model

## Key settings

| Setting | Value |
|--------|-------|
| Window size | 40 |
| Train/validation split | 80 / 20 engines |
| RUL cap | 125 cycles |

## Files

Put these beside the notebook:

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

Data source: NASA Prognostics Data Repository

https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

## Main model

The main LSTM is:

- `LSTM(64, return_sequences=True)`
- `Dropout(0.2)`
- `LSTM(32)`
- `Dropout(0.2)`
- `Dense(32, relu)`
- `Dense(1)`

Training uses Adam, MSE, and early stopping with `patience=5` and `restore_best_weights=True`.

## Extra models

- Random Forest on aggregated window features: mean, std, min, max, last value, and slope.
- Stacked RF + LSTM: appends the LSTM prediction as an extra feature.

## Performance snapshot

| Model | Val RMSE | Val MAE | Test RMSE | Test MAE |
|------|----------|---------|-----------|---------|
| Main LSTM | 12.02 | 9.00 | 14.77 | 11.09 |
| RF (agg features) | 13.05 | 9.97 | 13.45 | 9.81 |
| Stacked RF | 13.45 | 9.63 | 14.72 | 11.03 |

## What we tried

We also tested different hyperparameters, fixed vs decaying learning rates, RMSE-focused tuning, and simpler neural networks, but each of those setups produced higher error than the final two-layer LSTM.

## Reported results

The notebook also includes plots for training curves, validation predictions, test predictions, residuals, and model comparisons.

## Notes

The variance cell is diagnostic only; the actual drop step still uses the hardcoded `DROP_COLS` list. The scaler is fit on train only to avoid leakage, and the main evaluation model is the two-layer LSTM v2.
