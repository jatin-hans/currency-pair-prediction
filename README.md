# Which currency questions are actually answerable?

Ten years of EURUSD, USDJPY and USDINR. Three prediction targets — direction,
volatility, and next month's average rate — each graded against a named baseline
before any model was built.

All input data is stored in `data/` (prices from Yahoo Finance, rates and CPI
from FRED, positioning from the CFTC, FOMC statements), so everything runs
offline. The download scripts exist only to refresh it.

## Layout

```
notebooks/   the analysis, start here
scripts/     numbered pipeline — every table and chart regenerates from these
src/         the packaged feature/label/model pipeline the notebooks share
data/        raw and processed inputs, committed
outputs/     tables, figures and the dashboard payload
```

## Run it

```bash
uv sync
uv run pytest                              # the test suite is the contract
uv run python scripts/16_check_data.py     # offline validation of every stored file
uv run jupyter lab notebooks/
```

No trading strategy and no P&L anywhere: prediction quality against honest
baselines only. Nothing here is investment advice.

## License

MIT. See `LICENSE`.
