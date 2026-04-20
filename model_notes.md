# Model notes

## Why MAE is the headline metric

Delivery ETAs get shown to real users. If we promise 30 min and it takes 45, the user is angry — not 2.25× angrier than if it took 37. The pain scales roughly linearly with the miss, so MAE matches the business cost better than RMSE. RMSE would over-punish the occasional big miss, which in practice we can't avoid (traffic, weather). We still report RMSE and R² as sanity checks, but the leaderboard is sorted by MAE.

A secondary reason: MAE is in minutes. It's trivial to explain ("our ETA is off by ~6 min on average") to someone in ops. RMSE is in "minutes-ish" and always needs a caveat.

## What we tried

Three candidates on 5-fold CV, same preprocessing for all of them:

| Model | CV MAE | CV RMSE | CV R² |
|---|---|---|---|
| **LinearRegression** | **6.84 ± 0.45** | 11.02 | 0.75 |
| RandomForest (150 trees, depth 10) | 7.61 ± 0.53 | 11.81 | 0.72 |
| XGBoost (250 rounds, depth 5, lr 0.05) | 7.78 ± 0.73 | 12.20 | 0.70 |

Linear won — and it wasn't close.

Held-out test (200 rows): MAE **5.90**, RMSE 8.69, R² **0.83**.

## Why the simple model wins here

n = 1000 with ~4 useful signals. That's not enough data for trees to find real interactions without overfitting to noise. The relationship that matters (distance × traffic → time) is essentially linear once traffic is encoded as an ordinal, and we already baked that interaction in as a feature. Nothing left for a tree to discover except noise.

I expected RF/XGBoost to close some of the gap on the weather/time-of-day edges, but they lost on variance — the CV std on XGBoost (±0.73) is bigger than the gap between Linear's best and worst fold.

## Feature engineering that actually helped

Inside `FeatureEngineer`:
- Fill categorical NaNs with `"Unknown"` — the EDA showed ~30 nulls per cat column and "Unknown" carries signal (some of those missing rows have distinctive error patterns).
- Map Traffic_Level → {Low:1, Medium:2, High:3, Unknown:2}. Ordinal beats one-hot when the thing is literally ordered.
- `Distance_x_Traffic` interaction. A 10 km trip in high traffic isn't 10 km + penalty, it's ~3× slower per km.
- Bucket Courier_Experience into new/mid/experienced. The raw variable is noisy, the buckets capture the step-function shape visible in EDA.

All fitting (median imputation for courier experience) happens in `fit()`, so train/test leakage is avoided.

## Tuning approach

Light. Given n=1000, a proper grid search would just overfit the CV folds. So:
- Linear: no hyperparams to tune.
- RF and XGBoost: picked sensible defaults for a small tabular problem (shallow trees, moderate count). Sanity-checked with ±50% on n_estimators and depth — all within CV noise.

If we had 100× the data I'd run Optuna over XGBoost with 100 trials. Here it'd be theater.

## Known limitations

- **Linear on Distance is an approximation.** The error analysis shows MAE grows with distance (3.8 → 7.9 min from <3 km to 12+ km). There's probably a non-linear distance term (fixed overhead + per-km cost + congestion kicking in) that a spline would catch.
- **Only 1000 rows, one city.** Everything here is a pilot, not a production commitment.
- **No temporal features.** No day-of-week, no holiday, no historical restaurant load. If those columns show up in a richer dataset, trees would probably take over.

## Artifacts

- Trained pipeline: `outputs/models/best_model.pkl`
- Pipeline code: `model_pipeline/`
- Figures: `outputs/figures/`
