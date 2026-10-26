# Media Content Intelligence & Metadata Automation Platform

End-to-end ML pipeline that ingests media metadata (JSON/XML), validates it, builds NLP features, compares models, explains errors, tracks experiments, and serves genre classification with confidence-based manual-review routing.

> Independent portfolio project on **synthetic metadata** (50K generated titles). It shows the architecture and practices from my media-ML work; it contains no employer code or data.

## Pipeline
| Stage | Implementation |
|---|---|
| Ingestion & validation | JSON + XML loaders, rules for required fields / unique IDs / runtime range / known language (`app/ingest.py`) |
| NLP features | TF-IDF with 1-2 grams, sublinear TF, stop-words, vocab cap over title + synopsis + language |
| Model comparison | Logistic Regression vs XGBoost, 3-fold CV, macro-F1, majority-class baseline |
| Error analysis | confusion matrix + SHAP global term importance on the XGBoost model |
| Review routing | confidence threshold chosen so auto-accepted items reach ≥90% accuracy; the rest go to manual review |
| Tracking | MLflow params, CV and test metrics per model |
| Serving | FastAPI `/classify` (Pydantic validation, SQLAlchemy prediction log, confidence + per-class scores) |
| Delivery | PyTest, Dockerfile, GitHub Actions CI |

## Measured results (synthetic, deliberately noisy: 80% of words drawn from random genres)
- Majority-class baseline accuracy: 0.203
- Macro-F1: Logistic Regression 0.613, XGBoost 0.612 (3-fold CV 0.612 / 0.609); LogReg selected
- Review threshold 0.83: auto-accepts 20.5% of items at 90.2% accuracy, sends 79.5% to review. This is the honest trade-off at this noise level; with cleaner real metadata the auto-accept rate rises.
- Top SHAP terms: awkward, nature, blood, mishap, archive, funny, history, battle, monster, rescue

## Run it
```bash
pip install -r requirements.txt
python -m app.train 50000   # trains, logs to ./mlruns, writes artifacts/metrics.json
pytest -q
uvicorn app.main:app        # http://localhost:8000/docs
```
```bash
curl -s localhost:8000/classify -H 'content-type: application/json' -d '{"id":"X1","title":"Dark Cabin","synopsis":"haunted ghost killer dark scream curse cabin monster nightmare blood","language":"en","runtime_min":95}'
```
