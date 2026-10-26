import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import pytest
from fastapi.testclient import TestClient
from app.ingest import load_json, load_xml, make_records, validate


def test_validation_rejects_bad_rows():
    df = make_records(200); bad = df.iloc[[0]].copy(); bad["runtime_min"] = 9999
    import pandas as pd
    clean, rej = validate(pd.concat([df, bad, df.iloc[[1]]]))
    assert len(rej) >= 2 and clean.id.is_unique


def test_xml_and_json_loaders():
    x = "<r><item><id>A1</id><title>T</title><synopsis>" + "word " * 10 + "</synopsis><language>en</language><runtime_min>90</runtime_min></item></r>"
    assert load_xml(x).iloc[0].runtime_min == 90
    assert load_json('[{"id":"A","runtime_min":5}]').shape == (1, 2)


@pytest.fixture(scope="module")
def client():
    if not pathlib.Path("artifacts/genre.joblib").exists():
        from app.train import main; main(8000)
    from app.main import app
    return TestClient(app)


def test_classify_horror(client):
    r = client.post("/classify", json={"id": "X1", "title": "Dark Cabin", "synopsis": "haunted ghost killer dark scream curse cabin monster nightmare blood", "language": "en", "runtime_min": 95}).json()
    assert r["genre"] == "horror" and 0 <= r["confidence"] <= 1


def test_bad_language_422(client):
    r = client.post("/classify", json={"id": "X", "title": "T", "synopsis": "x" * 30, "language": "xx", "runtime_min": 90})
    assert r.status_code == 422
