"""12: score FOMC statements hawkish/neutral/dovish.

Dictionary scorer in the Apel-Grimaldi tradition: count hawkish and
dovish phrase hits per sentence, doc score = (hawk − dove) / (hawk + dove),
label at ±0.15. Deterministic and reproducible without API keys.
Output: data/raw/fomc/stance.parquet, outputs/case_study/tables/fomc_stance.csv,
plus sentence-level detail JSON.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHOD = "lexicon-v1"

HAWK = [
    "inflation.{0,40}elevated", "raise the target range", "raising the target",
    "tighten", "tightening", "restrictive", "upside risks to inflation",
    "inflation remains high", "price pressures", "strong(ly)? committed to returning inflation",
    "further firming", "firming of monetary policy", "solid pace", "strong job gains",
    "robust", "labor market remains strong",
]
DOVE = [
    "lower the target range", "lowering the target", "cut the target",
    "accommodative", "easing", "downside risks", "growth.{0,30}(slowed|moderated)",
    "job gains have slowed", "unemployment rate has (edged|moved) up",
    "weak", "soften", "patient", "muted inflation", "inflation.{0,30}below 2 percent",
    "shortfalls of employment", "supporting the flow of credit",
]


def score_doc(text: str) -> tuple[float, list[dict]]:
    sentences = [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if len(s.strip()) > 20]
    detail, hawk_n, dove_n = [], 0, 0
    for s in sentences:
        low = s.lower()
        h = sum(bool(re.search(p, low)) for p in HAWK)
        d = sum(bool(re.search(p, low)) for p in DOVE)
        hawk_n += h
        dove_n += d
        if h or d:
            detail.append({"sentence": s[:220], "hawk": h, "dove": d})
    total = hawk_n + dove_n
    score = 0.0 if total == 0 else (hawk_n - dove_n) / total
    return score, detail


def main() -> int:
    fomc_dir = ROOT / "data" / "raw" / "fomc"
    rows, sentences = [], {}
    for path in sorted(fomc_dir.glob("*.txt")):
        text = path.read_text()
        score, detail = score_doc(text)
        label = "hawkish" if score > 0.15 else "dovish" if score < -0.15 else "neutral"
        rows.append({"date": path.stem, "score": round(score, 3), "label": label,
                     "n_hits": sum(d["hawk"] + d["dove"] for d in detail),
                     "doc_type": "FOMC statement", "method": METHOD})
        sentences[path.stem] = detail

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").sort_index()
    df.to_parquet(fomc_dir / "stance.parquet")
    df.to_csv(ROOT / "outputs" / "case_study" / "tables" / "fomc_stance.csv")
    (fomc_dir / "stance_sentences.json").write_text(json.dumps(sentences, indent=1))
    print(df["label"].value_counts().to_dict())
    print(df.tail(6).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
