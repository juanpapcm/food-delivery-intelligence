# Food-delivery time intelligence

Technical assessment for a food-delivery platform. The project takes a 1,000-row sample of historical deliveries and builds an end-to-end ETA prediction system: exploratory analysis, a modular training pipeline comparing LinearRegression / RandomForest / XGBoost with 5-fold CV, a held-out evaluation with per-cohort error breakdown, and a FastAPI service that serves predictions with a confidence interval. SQL analyses for the same domain (top customer areas, courier performance, restaurant profitability) are included in [sql/](sql/).

## Repository structure

```
food-delivery-intelligence/
├── README.md
├── requirements.txt
├── data/
│   └── Food_Delivery_Times.csv
├── sql/
│   ├── sql_queries.sql           # 5 required queries
│   └── sql_insights.md           # additional business analyses
├── model_pipeline/
│   ├── config.py                 # paths, features, hyperparameters
│   ├── preprocessing.py          # FeatureEngineer + ColumnTransformer
│   ├── train.py                  # CV across candidates, fit + save best
│   ├── evaluate.py               # MAE / RMSE / R² + cohort error analysis
│   ├── predict.py                # load pipeline, predict from dict/df
│   ├── eda.py                    # EDA script (figures + console summary)
│   └── run_pipeline.py           # end-to-end orchestrator
├── api/
│   ├── app.py                    # FastAPI app (POST /predict, GET /health)
│   ├── schemas.py                # Pydantic request/response models
│   └── requirements.txt
├── scripts/
│   └── feature_importance.py     # builds feature importance + residual plots
├── outputs/
│   ├── figures/                  # EDA + explainability plots
│   ├── models/best_model.pkl     # trained pipeline artifact
│   └── reports/                  # CSV exports
├── EDA_report.md
├── model_notes.md                # modeling choices, metric rationale, tuning
├── explainability.md             # feature importance + residual analysis
├── error_insights.md             # where the model fails (by weather, traffic, …)
└── strategic_reflections.md      # 5 assessment questions
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended. `numpy<2` is pinned because some wheels (matplotlib in particular) were compiled against numpy 1.x in this environment.

## Run the EDA

```bash
python -m model_pipeline.eda
```

Writes 5 figures to `outputs/figures/` and prints the summary that powers [EDA_report.md](EDA_report.md).

## Run the training pipeline

```bash
python -m model_pipeline.run_pipeline
```

Does, in order: load → train/test split → 5-fold CV across the 3 candidates → fit best on full train set → save to `outputs/models/best_model.pkl` → evaluate on held-out test → print per-cohort error breakdown. Runs in a few seconds on a laptop.

## Run the API

```bash
uvicorn api.app:app --reload --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "distance_km": 5.2,
    "weather": "Rainy",
    "traffic_level": "High",
    "time_of_day": "Evening",
    "vehicle_type": "Scooter",
    "preparation_time_min": 15,
    "courier_experience_yrs": 2.5
  }'
```

Response:

```json
{
  "estimated_delivery_time_min": 50.11,
  "confidence_interval": [36.41, 63.81],
  "model_version": "1.0.0"
}
```

The confidence interval is ±1.28 × residual-std(train), i.e. roughly an 80% band under a normal-residual assumption.

## Main results

5-fold CV on the 800-row training set, held-out test on 200 rows.

| Model | CV MAE (min) | CV RMSE | CV R² | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|---|
| **LinearRegression** 🏆 | **6.84 ± 0.45** | 11.02 | 0.75 | **5.90** | **8.69** | **0.83** |
| RandomForest (150 × d10) | 7.61 ± 0.53 | 11.81 | 0.72 | - | - | - |
| XGBoost (250 × d5) | 7.78 ± 0.73 | 12.20 | 0.70 | - | - | - |

MAE is the headline metric - user-facing ETAs hurt roughly linearly in the miss size, so MAE matches the cost better than RMSE. Rationale in [model_notes.md](model_notes.md).

**Drivers (coef × feature_std, in minutes of ETA):**

| Feature | Std. effect | Direction |
|---|---|---|
| Distance_km | +17.1 | farther → slower |
| Preparation_Time_min | +7.0 | slower kitchen → slower delivery |
| Traffic_Ord (Low<Med<High) | +4.2 | more traffic → slower |
| Weather = Clear | −2.2 | clear weather faster |
| Courier_Experience_yrs | −1.8 | experience shaves ~2 min per SD |

Three physical features explain 83% of variance. Deep dive in [explainability.md](explainability.md).

**Where the model fails** (test set, by cohort):

| Cohort | MAE | Bias (actual − pred) |
|---|---|---|
| Foggy weather | 9.57 | **+5.51** (underestimates) |
| High traffic | 7.95 | **+2.45** (underestimates) |
| Distance 12+ km | 7.89 | −0.92 (noisier, not biased) |

Full tables and fixes in [error_insights.md](error_insights.md).

## Further reading

- [EDA_report.md](EDA_report.md) - data quality, distributions, correlations
- [model_notes.md](model_notes.md) - why MAE, why Linear won, tuning approach
- [explainability.md](explainability.md) - feature importance, residual analysis
- [error_insights.md](error_insights.md) - cohort error breakdown + fixes
- [strategic_reflections.md](strategic_reflections.md) - 5 questions (rainy-day bias, city transfer, productionization, signature insight)
- [sql/sql_queries.sql](sql/sql_queries.sql), [sql/sql_insights.md](sql/sql_insights.md)

## GenAI disclosure

GenAI tools were used only for:

- Documentation formatting and `.md` structure.
- Boilerplate generation (docstrings, type hints, initial file skeletons).
- Minor debugging and syntax checks.
- README drafting and polish.

All analysis, modeling logic, SQL queries, feature engineering, and strategic insights are my own. Every GenAI output was validated by running the code and adjusting to the problem context. See [strategic_reflections.md](strategic_reflections.md) for the fuller disclosure.
