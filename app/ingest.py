"""Metadata ingestion: synthetic generator + JSON/XML loaders + validation (no real media data is used)."""
import json, xml.etree.ElementTree as ET
import numpy as np, pandas as pd

GENRES = {
    "action": "explosion chase mission rescue fight agent heist battle weapon pursuit".split(),
    "romance": "love wedding heart couple kiss relationship date family reunion proposal".split(),
    "horror": "haunted ghost killer dark scream curse cabin monster nightmare blood".split(),
    "documentary": "history interview archive nature investigation footage expedition science culture real".split(),
    "comedy": "funny hilarious prank roommates awkward mishap sitcom quirky party chaos".split(),
}
COMMON = "story city young team journey secret night life new last".split()
LANGS = ["en", "es", "fr", "hi", "de"]


def make_records(n=50_000, seed=3, noise=0.8):
    rng = np.random.default_rng(seed); names = list(GENRES); rows = []
    for i in range(n):
        g = names[rng.integers(len(names))]
        words = [rng.choice(GENRES[g] if rng.random() > noise else GENRES[names[rng.integers(5)]]) for _ in range(rng.integers(8, 18))]
        words += list(rng.choice(COMMON, rng.integers(3, 8)))
        rng.shuffle(words)
        rows.append({"id": f"T{i:06d}", "title": " ".join(rng.choice(words, 3)).title(), "synopsis": " ".join(words),
                     "language": LANGS[rng.integers(5)], "runtime_min": int(rng.integers(20, 180)), "genre": g})
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame):
    """Return (clean, rejected). Rules: required fields, unique id, runtime range, known language."""
    bad = df.id.duplicated() | df.synopsis.fillna("").str.len().lt(20) | ~df.runtime_min.between(1, 600) | ~df.language.isin(LANGS)
    return df[~bad].copy(), df[bad].copy()


def load_json(text): return pd.DataFrame(json.loads(text))


def load_xml(text):
    root = ET.fromstring(text)
    return pd.DataFrame([{c.tag: c.text for c in item} | {"runtime_min": int(item.findtext("runtime_min", "0"))} for item in root.iter("item")])
