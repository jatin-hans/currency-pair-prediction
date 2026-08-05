"""Offline validation of stored data files: price/macro/COT/FOMC parquets
and dashboard JSON payloads. Exits 1 if any check fails."""

import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
problems = []


def fail(msg):
    problems.append(msg)


def report(name, df):
    print(f"{name} ok ({len(df)} rows, {df.index[0].date()}..{df.index[-1].date()})")


def check_index(name, df):
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        fail(f"{name}: index not sorted strictly increasing")


PRICE_RANGES = {
    "EURUSD_X": (0.8, 1.6), "INR_X": (55, 100), "JPY_X": (75, 165),
    "DX-Y_NYB": (70, 130), "_VIX": (8, 90),
}


def check_prices():
    for tick, (lo, hi) in PRICE_RANGES.items():
        name = f"data/raw/{tick}.parquet"
        df = pd.read_parquet(ROOT / name)
        if "close" not in df.columns:
            fail(f"{name}: no close column")
            continue
        check_index(name, df)
        c = df["close"]
        if c.isna().any():
            fail(f"{name}: {c.isna().sum()} NaN closes")
        elif (c <= 0).any():
            fail(f"{name}: non-positive closes")
        elif not c.between(lo, hi).all():
            fail(f"{name}: close outside [{lo}, {hi}] (min {c.min():.3f}, max {c.max():.3f})")
        gap = df.index.to_series().diff().max()
        if gap > pd.Timedelta(days=14):
            fail(f"{name}: calendar gap of {gap.days} days")
        if df.index[-1] < pd.Timestamp("2026-07-31"):
            fail(f"{name}: stale, last date {df.index[-1].date()}")
        report(name, df)


def check_macro():
    for path in sorted((ROOT / "data/raw/macro").glob("*.parquet")):
        if path.name.startswith("cot_"):
            continue
        name = f"data/raw/macro/{path.name}"
        df = pd.read_parquet(path)
        if list(df.columns) != ["value"]:
            fail(f"{name}: columns {list(df.columns)}, expected ['value']")
            continue
        check_index(name, df)
        # CPI floor is 20, not 80: US series starts 1947 (~21) and EZ uses a
        # non-2015 index base (~76 in 2015); both are valid index levels
        lo, hi = (-1, 20) if path.name.startswith("rate_") else (20, 400)
        v = df["value"].dropna()
        if not v.between(lo, hi).all():
            fail(f"{name}: value outside [{lo}, {hi}] (min {v.min()}, max {v.max()})")
        report(name, df)


def check_cot():
    for cur in ("EUR", "JPY"):
        name = f"data/raw/macro/cot_{cur}.parquet"
        df = pd.read_parquet(ROOT / name)
        if not {"lev_net", "am_net"} <= set(df.columns):
            fail(f"{name}: columns {list(df.columns)}")
        check_index(name, df)
        if (df.index.dayofweek != 4).any():
            fail(f"{name}: non-Friday dates present")
        gap = df.index.to_series().diff().max()
        if gap > pd.Timedelta(days=8):
            fail(f"{name}: weekly gap of {gap.days} days")
        if len(df) < 600:
            fail(f"{name}: only {len(df)} rows")
        report(name, df)


def check_fomc():
    txts = sorted(glob.glob(str(ROOT / "data/raw/fomc/*.txt")))
    for t in txts:
        n = len(Path(t).read_text())
        if n < 400:
            fail(f"data/raw/fomc/{Path(t).name}: only {n} chars")
    idx = pd.read_parquet(ROOT / "data/raw/fomc/index.parquet")
    if len(idx) != len(txts):
        fail(f"fomc/index.parquet: {len(idx)} rows vs {len(txts)} txt files")
    report("data/raw/fomc/index.parquet", idx)
    st = pd.read_parquet(ROOT / "data/raw/fomc/stance.parquet")
    if len(st) != len(idx):
        fail(f"fomc/stance.parquet: {len(st)} rows vs index {len(idx)}")
    if not st["score"].between(-1, 1).all():
        fail("fomc/stance.parquet: score outside [-1, 1]")
    bad = set(st["label"]) - {"hawkish", "neutral", "dovish"}
    if bad:
        fail(f"fomc/stance.parquet: bad labels {bad}")
    report("data/raw/fomc/stance.parquet", st)


def no_nan(_):
    raise ValueError("NaN/Infinity in JSON")


def check_payloads():
    for fn in ("dashboard_data.json", "turbulence_har.json", "period_average.json"):
        try:
            data = json.loads((ROOT / "outputs" / fn).read_text(), parse_constant=no_nan)
        except ValueError as e:
            fail(f"outputs/{fn}: {e}")
            continue
        print(f"outputs/{fn} ok")
        if fn != "dashboard_data.json":
            continue
        if data["as_of"] != "2026-07-31":
            fail(f"outputs/{fn}: as_of {data['as_of']}")
        for pair, pd_ in data["pairs"].items():
            for hz, h in pd_["horizons"].items():
                for mname, m in h["models"].items():
                    where = f"outputs/{fn} {pair}/{hz}/{mname}"
                    if len(m["recent"]) != 12:
                        fail(f"{where}: recent has {len(m['recent'])} entries")
                    ctx = m.get("context")
                    if ctx and not (ctx["hits"] <= ctx["active"] <= ctx["n"]):
                        fail(f"{where}: context hits/active/n inconsistent {ctx}")


def main():
    check_prices()
    check_macro()
    check_cot()
    check_fomc()
    check_payloads()
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    print("\nOK")


if __name__ == "__main__":
    main()
