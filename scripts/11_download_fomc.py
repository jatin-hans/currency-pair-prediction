"""11: download FOMC post-meeting statements 2015–2026.

Statement URLs follow monetaryYYYYMMDDa.htm; meeting dates come from the
Fed's calendar and historical-year pages. Raw text is stored per statement
(data/raw/fomc/YYYYMMDD.txt) plus an index parquet. Statements land ~2pm ET;
Yahoo stamps FX closes at the START of the UTC day, so a statement dated
day t is first usable at the FX close stamped t+1 — downstream joins must
lag one day. RBI statements are not downloaded: the portal is JS-driven
and could not be enumerated reliably.
"""

from __future__ import annotations

import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://www.federalreserve.gov"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; regime-lab case study)"}


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


class TextExtract(HTMLParser):
    """Collect visible text from the statement's article region."""

    def __init__(self):
        super().__init__()
        self.chunks: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer", "header"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer", "header") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.chunks.append(data.strip())


def statement_text(html: str) -> str:
    # The statement body sits between the title block and the voting/
    # implementation-note boilerplate; keep it simple: all visible text,
    # then trim to the segment starting at the first real paragraph.
    p = TextExtract()
    p.feed(html)
    text = "\n".join(p.chunks)
    start = text.find("Recent indicators")
    if start < 0:
        start = text.find("Information received")
    if start < 0:
        start = text.find("The Federal Open Market Committee")
    end = text.find("Voting for the")
    if end < 0:
        end = len(text)
    return text[max(start, 0):end].strip()


def main() -> int:
    out_dir = ROOT / "data" / "raw" / "fomc"
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = [f"{BASE}/monetarypolicy/fomccalendars.htm"] + [
        f"{BASE}/monetarypolicy/fomchistorical{y}.htm" for y in range(2015, 2022)
    ]
    dates: set[str] = set()
    for page in pages:
        try:
            html = fetch(page)
        except Exception as exc:  # noqa: BLE001
            print(f"  calendar {page}: skipped ({exc})")
            continue
        dates |= set(re.findall(r"monetary(\d{8})a\.htm", html))
        time.sleep(0.4)

    rows = []
    for d in sorted(dates):
        if not (2015 <= int(d[:4]) <= 2026):
            continue
        path = out_dir / f"{d}.txt"
        if path.exists():
            rows.append({"date": d, "n_chars": path.stat().st_size})
            continue
        try:
            text = statement_text(fetch(f"{BASE}/newsevents/pressreleases/monetary{d}a.htm"))
        except Exception as exc:  # noqa: BLE001
            print(f"  {d}: skipped ({exc})")
            continue
        if len(text) < 400:
            print(f"  {d}: skipped (extracted only {len(text)} chars)")
            continue
        path.write_text(text)
        rows.append({"date": d, "n_chars": len(text)})
        time.sleep(0.4)

    idx = pd.DataFrame(rows)
    idx["date"] = pd.to_datetime(idx["date"], format="%Y%m%d")
    idx = idx.set_index("date").sort_index()
    idx.to_parquet(out_dir / "index.parquet")
    print(f"{len(idx)} statements, {idx.index[0].date()} → {idx.index[-1].date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
