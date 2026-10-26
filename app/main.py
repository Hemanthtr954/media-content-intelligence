import json, logging, time
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import Column, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from .ingest import LANGS

logging.basicConfig(level=logging.INFO); log = logging.getLogger("media")
Base = declarative_base(); engine = create_engine("sqlite:///predictions.db", connect_args={"check_same_thread": False}); Session = sessionmaker(engine)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True); title_id = Column(String(20)); label = Column(String(30)); confidence = Column(Float); needs_review = Column(Integer); detail = Column(Text)


Base.metadata.create_all(engine)
app = FastAPI(title="Media Content Intelligence"); _b = {}


def bundle():
    if not _b: _b.update(joblib.load("artifacts/genre.joblib"))
    return _b


class Item(BaseModel):
    id: str = Field(min_length=1, max_length=20); title: str = Field(min_length=1)
    synopsis: str = Field(min_length=20); language: str; runtime_min: int = Field(ge=1, le=600)


@app.get("/health")
def health(): return {"status": "ok"}


@app.post("/classify")
def classify(item: Item):
    if item.language not in LANGS: raise HTTPException(422, f"language must be one of {LANGS}")
    b = bundle(); t = time.perf_counter()
    p = b["model"].predict_proba(b["vec"].transform([f"{item.title} {item.synopsis} {item.language}"]))[0]
    i = int(p.argmax()); label, conf = str(b["le"].classes_[i]), float(p[i]); review = conf < b["threshold"]
    with Session() as s:
        s.add(Prediction(title_id=item.id, label=label, confidence=conf, needs_review=int(review), detail=json.dumps({"ms": round((time.perf_counter() - t) * 1000, 2)}))); s.commit()
    log.info(json.dumps({"id": item.id, "label": label, "conf": round(conf, 3), "review": review}))
    return {"id": item.id, "genre": label, "confidence": round(conf, 4), "needs_manual_review": review,
            "scores": {str(c): round(float(x), 4) for c, x in zip(b["le"].classes_, p)}}
