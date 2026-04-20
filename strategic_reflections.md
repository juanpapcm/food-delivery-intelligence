# Strategic reflections

Written as if presenting to Ops leadership after the initial build. Less "model report", more "here's what I'd actually do next".

---

## 1. The model underestimates delivery time on rainy days. Fix the model, the data, or the business expectations?

All three. In that order, but none of them in isolation.

**First, the model.** Rainy-day bias is structural — rain doesn't add a flat 2 minutes, it compounds with distance and traffic. A linear `Weather_Rainy` dummy can't express that. I'd add interaction features (`Weather_Rainy × Distance_km`, `Weather_Rainy × Traffic_High`) and, if we have the intensity, graduate from a binary "Rainy" flag to mm/h of precipitation. This alone should close most of the cohort-level bias we see in `error_insights.md`.

**Then, the data.** The current weather column is one word. Real rain has gradients — drizzle is nothing, a monsoon shuts the city. If we can pipe in hourly precipitation from a weather API and aggregate over the delivery window, we'll stop lumping very different situations into the same bucket. Same for Traffic — a live traffic index would beat our 3-level categorical.

**Then, expectations.** This is where I'd push hardest with Ops. Even a perfect model has a wider uncertainty band on rainy days. Instead of showing a single ETA, we should show a range, or bias the displayed number toward the 65th–70th percentile of our predicted distribution. The cost is ~2–3 minutes of "optimistic ETA" we give up; the win is fewer late-delivery complaints on exactly the days when customer patience is already thin. This is a product decision, not a model decision, but data has to make the case.

The honest answer is: you can't fix "the model is wrong on rainy days" by only fixing the model. The user still has a number in front of them; we just have to choose which number is least likely to disappoint.

---

## 2. Model works in Mumbai, deploying in São Paulo. How do we ensure generalization?

Short answer: assume it won't generalize, and plan around that assumption.

**Retrain on local data before launch.** The Mumbai model is a prior, not a product. Courier networks, road geometry, traffic patterns, weather distributions, restaurant prep cultures — all different. Even if the columns match, the coefficients won't. Target: 4–6 weeks of São Paulo data before the model goes to any user.

**Separate what's city-agnostic from what isn't.**
- Probably portable: the *shape* of the relationship (distance is linear-ish, traffic is convex, prep time adds one-for-one). These are physics / queueing facts.
- Definitely not portable: the *coefficients*. Mumbai's "minutes per km" is not São Paulo's.
- Unknown: which features matter. São Paulo might have features Mumbai doesn't (hills, rain seasonality), and some Mumbai features might be dead weight.

**Bootstrap the launch.** While we collect São Paulo data, run a hybrid ETA: a simple distance × speed baseline with a conservative buffer, *not* the Mumbai model. Launching a foreign model is worse than launching a dumb one.

**A/B test before full rollout.** Once the São Paulo-trained model exists, run it against the heuristic baseline on 10% of deliveries for 2 weeks. Primary metric: MAE on completed deliveries. Guardrail: customer-facing "late" rate. Only flip to 100% if both move in the right direction.

**Drift monitoring, day one.** Track feature distributions (distance, prep time, weather mix) and prediction distributions weekly. São Paulo's reality will shift — new neighborhoods go live, courier fleet changes, rainy season arrives. We need to see that coming, not learn about it from complaints.

---

## 3. GenAI disclosure

I used GenAI tools in a limited, clearly bounded way during this assessment:

- **Documentation formatting and structure** — tightening the layout of the `.md` files so they read consistently.
- **Boilerplate generation** — docstrings, type hints, initial file skeletons.
- **Minor debugging and syntax checks** — spotting small issues faster than flipping to docs.
- **README drafting and polish** — wording and section ordering on the README.

Everything substantive is my own work: the analysis, the modeling logic, the SQL queries, the feature engineering choices, and the strategic insights. I validated every GenAI output by running the code and adjusting to fit the problem context — nothing was accepted without being executed or cross-checked against real results.

---

## 4. Signature insight

The thing I didn't expect: **Distance, Prep Time, and Traffic together explain ~83% of variance (R² 0.83 on test).** That's the headline, but the interesting part is what it means — delivery time in this dataset is almost mechanical. There's very little "courier-specific" or "restaurant-specific" magic left to extract once you know the three physical facts of the trip.

Two implications stakeholders should care about:

**(a) Prep time is the only endogenous lever.** Distance is physics — you can't shorten it without moving warehouses. Traffic is the city's problem. Prep time is the restaurant's problem, and that's *our network's* problem. Every minute of kitchen wait is a minute added to the customer ETA, nearly one-for-one. If I were prioritizing operational investment based on this model, I'd put it into prep-time tooling: kitchen capacity alerts, order-batching for prep-time smoothing, restaurant-side SLA enforcement. The model is implicitly telling us the biggest fixable number is not "dispatch smarter" — it's "cook faster".

**(b) Courier experience matters less than I thought.** Going from a new courier to an experienced one saves about 2 minutes on average, holding everything else constant. That's real but modest. The business implication: training programs aren't going to transform delivery times. Experience is nice to have, not a growth lever. If we're wondering whether to invest in courier education vs. restaurant prep-time infrastructure, the model points pretty strongly at the latter.

The non-obvious finding isn't a hidden interaction or a clever feature — it's that **the boring features are the right features**, and the operational conclusion is correspondingly unglamorous: make the kitchens faster.

---

## 5. Going to production

Sketch of what I'd build. Scope is deliberately "enough to run safely", not "enterprise-grade from day one".

### Serving layer
- **FastAPI** wrapping `predict.py` with one endpoint: `POST /predict` taking the raw feature dict, returning `{eta_minutes, eta_low, eta_high, model_version}`. Pydantic schemas at the boundary.
- **Dockerized** with a slim Python base + the pipeline pickle baked into the image. Image tag = model version. No mutable state in the container.
- **Health** and **readiness** endpoints that actually load the model — catches a broken artifact at rollout, not at first request.

### CI/CD
- **GitHub Actions**: on PR, run `pytest` + `ruff` + a tiny smoke test that loads the model and predicts one row.
- On merge to `main`: build image, push to registry, tag with commit SHA.
- Deployment via whatever the platform already uses — I wouldn't invent a new system for this.
- Separate pipeline for *training* vs *serving*. Training is a scheduled job, serving is a service. They share a model artifact, not a process.

### Model registry
- **MLflow** (or the team's existing equivalent). Every training run logs params, metrics, the pipeline artifact, and a data-hash of the training set.
- Registry has `staging` and `production` tags. Rollouts are tag flips, not rebuilds.
- Rollback is a tag flip in the other direction.

### Monitoring
- **Evidently** (or similar) on a scheduled job, daily:
  - Feature drift: PSI on Distance, Traffic_Level, Weather, Prep_Time, etc. Alert if any feature's PSI > 0.2.
  - Prediction drift: distribution of ETAs week-over-week.
  - Data quality: null-rate per column.
- **Live MAE tracking**: once the actual delivery time lands (maybe 1h after prediction), we join back and compute rolling MAE. This is the most important monitor — the model can silently degrade while feature distributions look fine.
- **Alert thresholds**:
  - Hard alert: rolling 7-day MAE > 8 minutes (vs current ~6).
  - Soft alert: any single-day MAE > 10 minutes, or drift PSI > 0.2 sustained over 3 days.
  - Alerts go to a #model-health Slack channel with context, not just a ping.

### Retraining
- **Scheduled retrain** weekly, on the rolling last 90 days of data. Automatic only if: (a) CV MAE doesn't regress vs current production model, (b) no feature has extreme drift vs training set, (c) the validation cohort (most recent 7 days held out) improves or stays flat. Otherwise, human-in-the-loop.
- **Manual retrain trigger** for incidents — on-call DS can run the retrain job with a one-line command.

### A/B testing
- Feature-flagged per-delivery routing between model A (current prod) and model B (candidate). Start at 1%, ramp to 50/50 over ~10 days if nothing explodes.
- Primary: MAE on completed deliveries. Secondary: "late" rate (actual > shown ETA + buffer). Guardrail: zero increase in customer complaints tagged "ETA wrong".
- Minimum test duration: 2 weeks, to see a rainy/weekend/weekday cycle.

### What I'd *not* build on day one
- A feature store. Overkill for 4 features.
- Real-time inference autoscaling. A single container handles this volume.
- Full lineage tracking across model / feature / data. Worth it later, not before we have two models.

The main principle: the monitoring and the rollback are more important than the model itself. A slightly worse model you can see and fix beats a slightly better model you can't.
