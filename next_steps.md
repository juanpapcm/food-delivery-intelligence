# Next steps

Roadmap for the delivery-time system beyond this assessment.

## 1. Model improvements

- **New features**: exact hour-of-day (cyclic encoding), day-of-week, is_holiday, distance to nearest dark-kitchen / hub, courier recent-load, restaurant rolling prep-time.
- **Better interactions**: `Weather × Distance`, `Traffic × Distance` — the error analysis points straight at these as fixes for the Foggy / High-traffic underestimation.
- **Ensemble stacking**: Linear + XGBoost blended with a meta-learner. Linear handles the bulk; boosting handles the non-linear edges once we have the data to support it.
- **Hyperparameter tuning** with Optuna (50–100 trials) once we're past n ~ 10k rows. Not worth it at n=1000.
- **Quantile regression** (0.2 / 0.5 / 0.8) for honest confidence bands instead of the current normal-residual approximation — the residual fan shows the current CI is too narrow on long trips and too wide on short ones.

## 2. Data pipeline

- **Streaming ingest** via Kafka → delivery events land in the warehouse seconds after completion, not daily batches.
- **Feature store** (Feast) so training and serving read the same feature definitions, in the same units, same nulls handling — eliminates a whole class of silent skew.
- **Data-quality checks** with Great Expectations on each ingest: schema, ranges, null-rates, referential integrity on `restaurant_id` / `delivery_person_id`. Block loads on hard failures, alert on soft ones.
- **Scheduled retraining**: weekly job on rolling 90 days; auto-promote only if CV MAE doesn't regress and no feature has extreme drift vs training.

## 3. Production hardening

- **Health & readiness** endpoints that actually load the model (already in `api/app.py`). Liveness separate, so a stuck request doesn't trigger a rollout.
- **Rate limiting** at the gateway (per-client, per-minute) — protects the service from a misbehaving caller.
- **Caching** on high-frequency identical requests (same restaurant + customer area + traffic bucket) with a short TTL — cheap win on QPS.
- **Structured JSON logging** with request_id, latency, input hash, prediction — feeds into observability.
- **Kubernetes autoscaling** on p95 latency + CPU. Stateless service, safe to scale horizontally.
- **MLflow registry** for model versioning; rollouts are tag flips between `staging` and `production`, rollback is one flip back.

## 4. Business impact

- **A/B test** the model ETA vs the current rule-based ETA at 10% traffic for 2 weeks. Primary: MAE on completed deliveries. Guardrail: customer "late" complaint rate.
- **Live Ops dashboard**: current MAE by hour, drift indicators, cohort breakdowns — the error-analysis output from `evaluate.py` but live.
- **Alerting**: Slack alert when rolling 7-day MAE crosses 8 min or any feature PSI > 0.2 sustained 3 days. Routed to the DS on-call channel.
- **Feedback loop**: actual delivery time joins back to prediction ~1 hr post-delivery; rolling error feeds the monitoring dashboard and the retraining trigger. Closes the loop from prediction → reality → improvement.
