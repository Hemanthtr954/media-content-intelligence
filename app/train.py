"""Compare TF-IDF(1-2gram)+LogReg vs TF-IDF+XGBoost with CV, tune a review threshold, inspect errors, log to MLflow."""
import json, pathlib, sys
import joblib, mlflow, numpy as np, shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from .ingest import make_records, validate

ART = pathlib.Path("artifacts")


def main(n=50_000):
    df, rej = validate(make_records(n)); le = LabelEncoder(); y = le.fit_transform(df.genre)
    text = df.title + " " + df.synopsis + " " + df.language
    Xtr_t, Xte_t, ytr, yte = train_test_split(text, y, test_size=0.2, stratify=y, random_state=0)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=400, sublinear_tf=True, stop_words="english")
    Xtr, Xte = vec.fit_transform(Xtr_t), vec.transform(Xte_t)
    baseline = (np.bincount(ytr).argmax() == yte).mean()
    mlflow.set_experiment("media-genre"); out = {}
    cands = {"logreg": LogisticRegression(max_iter=300, C=3),
             "xgboost": XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, n_jobs=-1)}
    for name, m in cands.items():
        with mlflow.start_run(run_name=name):
            cv = cross_val_score(m, Xtr, ytr, cv=3, scoring="f1_macro")
            m.fit(Xtr, ytr); pred = m.predict(Xte); f1 = f1_score(yte, pred, average="macro")
            mlflow.log_params({"model": name, "vocab": 400, "ngram": "1-2"})
            mlflow.log_metrics({"cv_f1": cv.mean(), "test_f1": f1})
            out[name] = {"cv_f1": round(float(cv.mean()), 4), "test_f1": round(float(f1), 4), "model": m}
    best = max(out, key=lambda k: out[k]["test_f1"]); model = out[best]["model"]
    proba = model.predict_proba(Xte); conf = proba.max(1); pred = proba.argmax(1)
    # choose confidence threshold: accuracy of auto-accepted items >= 0.9
    thr = next((t for t in np.linspace(0.3, 0.95, 66) if (pred[conf >= t] == yte[conf >= t]).mean() >= 0.90), 0.9)
    auto = conf >= thr
    cm = confusion_matrix(yte, pred).tolist()
    # SHAP on a sample for the selected tree model (global term importance)
    # SHAP (global term importance) on the XGBoost candidate, so errors can be traced to terms
    xgb = out["xgboost"]["model"]
    sv = np.abs(np.array(shap.TreeExplainer(xgb).shap_values(Xte[:300].toarray())))
    sv = sv.mean(axis=tuple(i for i in range(sv.ndim) if sv.shape[i] != len(vec.get_feature_names_out()) or i == 0))
    top_terms = [str(vec.get_feature_names_out()[i]) for i in np.argsort(-sv)[:10]]
    ART.mkdir(exist_ok=True)
    joblib.dump({"vec": vec, "model": model, "le": le, "threshold": float(thr)}, ART / "genre.joblib")
    res = {"records_in": n, "valid": len(df), "rejected": len(rej), "majority_baseline_acc": round(float(baseline), 3),
           "models": {k: {a: b for a, b in v.items() if a != "model"} for k, v in out.items()}, "selected": best,
           "review_threshold": float(thr), "auto_accept_rate": round(float(auto.mean()), 3),
           "auto_accept_accuracy": round(float((pred[auto] == yte[auto]).mean()), 3),
           "manual_review_rate": round(float(1 - auto.mean()), 3), "confusion_matrix": cm, "top_terms": top_terms}
    (ART / "metrics.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50_000)
