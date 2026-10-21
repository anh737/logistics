# SwiftRoute Logistics — Delivery Delay Prediction

DSAI mini-coursework: predict at pickup which shipments will be delivered outside their promised window, so that Operations can intervene on the highest-risk 15 %.

The completed analysis is in [coursework_completed.ipynb](coursework_completed.ipynb); the saved model is [model.pkl](model.pkl). The untouched starter notebook is kept as [coursework.ipynb](coursework.ipynb).

## Files

| File | Purpose |
|---|---|
| `coursework_completed.ipynb` | Assessed notebook: written answers, EDA, cleaning and joins, feature engineering, models, evaluation, business impact, monitoring plan and both extension tasks |
| `coursework.ipynb` | Original starter notebook (unchanged) |
| `model.pkl` | Selected model saved with joblib: median imputer → standard scaler → logistic regression |
| `model_features.json` | Feature list (in order), promise-day mapping and training-data description for scoring |
| `data/` | Supplied `shipments.csv`, `routes.csv`, `carriers.csv`, `weather_events.csv` |
| `figures/` | Figures exported from the notebook outputs |
| `business_brief.md`, `data_dictionary.md` | Supplied business context and data definitions |
| `requirements.txt` | Pinned dependencies of the environment used to run the notebook (Python 3.13) |

## Method

- Data quality checks: 25 re-keyed duplicate shipments removed; missing weight and declared value imputed with training-set medians inside the pipeline, with missing-value indicators kept as features; referential integrity between the four tables verified.
- Joins: route and carrier attributes joined many-to-one without changing the row count; weather matched by region and inclusive date range on the pickup date, with the maximum severity of overlapping events taken separately for the origin and destination region.
- Features: weather severity and flags, calendar (month, winter, weekday), parcel transforms and missing indicators, carrier × route interaction, promised days minus typical transit time, distance per hop, one-hot service tier. All features use information known at pickup.
- Models: logistic regression (L2, balanced class weights) and a depth-limited random forest, both with preprocessing inside a scikit-learn `Pipeline`, compared by five-fold stratified cross-validation on the 8,000 training shipments and confirmed on 2,000 held-out validation shipments (stratified 80/20 split, `random_state=42`).
- Evaluation: ROC curves, score distributions, coefficients and permutation importance; delay capture rate and precision for the top-15 % policy, decile lift, and a time-based check (train 2024–2025, test 2026 H1).
- Extensions: k-means clustering of routes with elbow and silhouette diagnostics; SHAP explanations of the ten highest-risk validation shipments.

## Results

| Measure | Logistic regression (selected) | Random forest |
|---|---:|---:|
| Mean CV ROC-AUC ± fold SD | 0.868 ± 0.017 | 0.856 ± 0.016 |
| Validation ROC-AUC | 0.869 | 0.863 |
| Train–validation gap | +0.004 | +0.049 |
| ROC-AUC on 2026 H1 (trained on 2024–2025) | 0.843 | 0.826 |
| Delays captured in the top 15 % (300 of 2,000 validation shipments) | 149 of 238 (62.6 %) | 148 of 238 (62.2 %) |

Flagging the top 15 % by model score finds about 4.2 times as many delays as flagging 15 % at random. At an assumed £45 saved per avoided delay this is £6,705 versus £1,606 on the validation set, and an incremental gross benefit of roughly £217,000 per month at 85,000 shipments if every captured delay were prevented (about £108,000 at a 50 % prevention rate), before intervention costs.

## Reproduce

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
uv run jupyter lab
```

Open `coursework_completed.ipynb`, select the kernel from this `.venv`, and run all cells. Running the notebook regenerates `model.pkl` and `model_features.json`. Without `uv`, `python3.13 -m venv .venv` and `.venv/bin/pip install -r requirements.txt` work the same way.

`model.pkl` expects the engineered feature table listed in `model_features.json` (31 numeric columns), not the raw CSVs; the joins and feature construction are implemented in Sections 4–5 of the notebook.

## Notes

- The data files are the supplied coursework data and are included only for reproducibility.
- The random validation split does not establish performance on future periods or entirely new routes; the 2026 time-based check is reported as a supplementary indication.
