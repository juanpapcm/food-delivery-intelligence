# Error insights

Held-out test (n=200). Overall MAE **5.90 min**, bias ~0.
But "overall" hides where the model actually hurts users. Breaking it down:

## By weather

| Weather | n | MAE | Bias (actual − pred) |
|---|---|---|---|
| **Foggy** | 21 | **9.57** | **+5.51** |
| Rainy | 40 | 6.42 | −1.21 |
| Snowy | 20 | 6.14 | −1.37 |
| Windy | 16 | 6.03 | +1.46 |
| NaN | 8 | 5.61 | −1.99 |
| Clear | 95 | 4.83 | −2.00 |

**Foggy is the big flag.** 62% higher error than average, and crucially the bias is +5.5 min - **the model systematically underestimates foggy deliveries by ~5.5 min**. So ETAs during fog are late, not just noisy. This is a user-facing problem: we'd be promising 35 min and delivering at 40+.

Two things going on, I think. (1) Fog is rare (21/200 = 10%) so the linear fit is dominated by the other weathers. (2) Fog probably interacts with distance more than other weathers do - a linear term can't capture "fog adds ~3 min on short trips and ~8 min on long trips". A fix would be a `Weather_Foggy × Distance_km` feature or switching to a quantile model for the tail.

Clear and Rainy are well-calibrated (bias near zero or slightly optimistic, which is fine - users prefer over-delivering vs under-delivering).

## By traffic

| Traffic | n | MAE | Bias |
|---|---|---|---|
| **High** | 38 | **7.95** | **+2.45** |
| Low | 72 | 5.95 | −1.90 |
| Medium | 79 | 5.09 | −1.37 |
| NaN | 11 | 4.41 | +0.78 |

Same story, different feature. **High traffic underestimated by 2.5 min.** The ordinal encoding is pulling a linear penalty (~4 min from Low→High, per the explainability doc), but the reality is probably convex - High isn't 2× Medium, it's more like 2.5-3× during the actual congested windows. Low/Medium are slightly overestimated for the mirror reason.

The combination Foggy + High is the worst-case cohort the error analysis can't show (too few rows in the test set), but based on the marginals, it'd likely be +8 min underestimated.

## By distance bucket

| Distance | n | MAE | Bias |
|---|---|---|---|
| <3km | 25 | 3.83 | +0.71 |
| 3-7km | 43 | 4.47 | −0.17 |
| 7-12km | 55 | 5.18 | −1.51 |
| **12+km** | 77 | **7.89** | −0.92 |

Errors grow monotonically with distance - 2× the error on 12+km vs <3km. Bias stays roughly zero, so it's a **variance** problem, not a **miscalibration** problem: long trips are inherently noisier. Matches the residual fan in the residuals plot.

Actionable read: ETAs on 12+km trips should come with a wider uncertainty band. If we show a single number, we'll keep burning those users.

## By time of day

| Time_of_Day | n | MAE | Bias |
|---|---|---|---|
| **NaN** | 6 | **14.54** | −3.95 |
| Afternoon | 60 | 6.52 | −0.61 |
| Evening | 57 | 5.36 | −1.60 |
| **Night** | 9 | 5.29 | **+4.29** |
| Morning | 68 | 5.13 | −0.45 |

Two tiny-n observations with big bias:
- **NaN rows** (n=6): MAE 14.5 - our "Unknown" fallback isn't carrying enough signal. These are likely systematically different rows (maybe data capture issues in a specific time slot); worth investigating upstream rather than fixing in the model.
- **Night** (n=9): bias +4.3. We underestimate night deliveries. Could be thin staff, lower courier density. Sample is small so don't over-read it, but worth a targeted look.

Morning / Evening / Afternoon are all fine.

## By vehicle

| Vehicle | n | MAE | Bias |
|---|---|---|---|
| Bike | 106 | 6.23 | −0.44 |
| Scooter | 62 | 6.02 | −0.97 |
| Car | 32 | 4.59 | −1.14 |

Car trips are best-predicted - likely a cleaner subset (fewer stuck-in-bike-lane surprises). No strong bias issues. Not actionable.

## Where the model fails, in one paragraph

**The model is late on the cohorts that matter most to users: Foggy weather (+5.5 min) and High traffic (+2.5 min).** These are exactly the conditions where customer expectations are already fragile and a late ETA hurts the most. Both failures have the same structural cause - linear terms can't capture the convex penalty that adverse conditions impose, especially combined with distance. The fix is either (a) an interaction term like `Bad_Weather × Distance` + `High_Traffic × Distance`, (b) a quantile model that shifts ETAs to the 60th percentile under adverse conditions (small cost to most users, big win to the tail), or (c) a more flexible model once we have more than 1000 rows.

On top of that, the model's **uncertainty grows with distance**: 12+km trips have 2× the MAE of <3km. A single-point ETA is dishonest for long trips - showing a range would be more truthful.

## Suggested next experiments

1. Add `Foggy × Distance_km` and `High_Traffic × Distance_km` interaction features → should close 30-50% of the cohort-level bias.
2. Quantile regression at 0.6 or 0.7 → shifts ETAs slightly late-biased by design, reduces user-facing "late" complaints.
3. Per-distance-bucket uncertainty bands shown in product.
4. Dig into the NaN Time_of_Day rows - looks like an upstream data problem, not a modeling one.
