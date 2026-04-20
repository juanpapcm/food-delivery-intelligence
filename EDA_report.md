# EDA Report

Dataset: `data/Food_Delivery_Times.csv` — 1,000 rows, 9 columns.
Reproduce with `python model_pipeline/eda.py`. Figures in [outputs/figures/](outputs/figures/).

---

## Target

`Delivery_Time_min`: mean **56.7**, median **55.5**, std 22.1, range [8, 153],
skew **0.51**. Slight right tail but close to symmetric — fine to model on the
raw scale; a log transform is not justified at this skew.
→ [01_target_distribution.png](outputs/figures/01_target_distribution.png)

## What drives delivery time

| Feature | Signal | Note |
|---|---|---|
| `Distance_km` | **r = 0.78** (p ≈ 0) | Dominant driver. Tight linear relationship, no obvious saturation. |
| `Weather` | median spread **14 min** | Snowy (66) > Rainy (60) > Foggy/Windy > Clear (52). |
| `Traffic_Level` | median spread **14 min** | High (65) > Medium > Low (51). Monotonic, treat as ordinal. |
| `Preparation_Time_min` | r = 0.31 | Additive on top of distance — mostly kitchen time pass-through. |
| `Time_of_Day` | spread 4 min | Weak effect once distance/weather are controlled. |
| `Vehicle_Type` | spread 1.5 min | Effectively noise in this sample. |
| `Courier_Experience_yrs` | r = -0.09 | No usable signal. |

→ [02_distance_vs_time.png](outputs/figures/02_distance_vs_time.png),
  [03_boxplots_categorical.png](outputs/figures/03_boxplots_categorical.png),
  [05_correlation_heatmap.png](outputs/figures/05_correlation_heatmap.png)

Takeaway: a single numeric feature (distance) plus two ordinals (weather,
traffic) should carry most of the predictive performance. Vehicle type and
courier experience are candidates for dropping unless the model surfaces
interactions.

## Outliers

IQR method on the target:
- `Delivery_Time_min`: 6 rows (0.6%) above 116 min. Plausible — all pair long
  distance with bad weather or high traffic, not data errors.
- `Distance_km`: **zero** outliers by IQR; the longest trip (≈20 km) is
  within bounds.

→ [04_iqr_outliers.png](outputs/figures/04_iqr_outliers.png)

**Treatment.** Keep the 6 tail rows in training. They are operationally
realistic (storms + long trips are exactly the cases Ops cares about) and
removing them would bias the model toward the easy regime. For robustness
we'll use tree-based models in the main pipeline, which are resistant to
tail influence; if we add a linear baseline we'll report metrics with and
without the tail as a sensitivity check.

## Missing values

3% missing in `Weather`, `Traffic_Level`, `Time_of_Day`, and
`Courier_Experience_yrs` — same count (30) across four columns, which
suggests a shared logging gap rather than feature-specific failure.

**Treatment.**
- Categoricals → explicit `"Unknown"` category. Imputing to the mode would
  hide the logging issue from the model and from monitoring later.
- `Courier_Experience_yrs` → median imputation, with a `_was_missing` flag.
  Given its near-zero correlation with the target we don't expect this to
  matter, but the flag keeps the door open for tree splits on "missing".

## Modeling assumptions

1. **Distance is the dominant signal.** Any model worse than a distance-only
   baseline (expected R² ≈ 0.6 from r=0.78) is broken.
2. **Weather and Traffic are ordinal**, not just categorical. Encoding them
   with an explicit order (Clear < Foggy/Windy < Rainy < Snowy;
   Low < Medium < High) is a cheap lever for linear models and a tie-breaker
   for trees.
3. **The 30-row missingness is MCAR-ish** (same count in 4 columns points at
   a logging gap) — safe to impute without modeling the missingness process.
4. **Weak features stay in for now.** `Vehicle_Type`, `Time_of_Day`,
   `Courier_Experience_yrs` have low marginal signal, but dropping them
   pre-modeling would miss potential interactions (e.g. bike + rain).
   Drop decisions deferred to feature-importance / ablation.
5. **Sample size is small (n=1000).** We'll prefer models that regularize
   well (gradient boosting with shallow trees, ridge baseline) over
   high-capacity ones, and validate with k-fold rather than a single split.
6. **No temporal split is possible** — the CSV has no timestamp. Random
   k-fold is acceptable here, but in production this becomes a time-based
   split to catch drift.
