# Which currency questions are actually answerable?

Ten years of EURUSD, USDJPY and USDINR. Three prediction targets, every model
graded against a named baseline before it was built. Two of the three questions
turned out to be dead ends. The third improves a rate that gets set every month.

| # | Question | Verdict |
|---|---|---|
| 01 | Where does the rate go next? *(direction — next day, week or month)* | **No edge** |
| 02 | How big will the moves be? *(volatility — next week)* | **Error halved** |
| 03 | What rate should we fix ahead? *(next month's average)* | **16–28% more accurate** |

The full analysis lives in two notebooks:
[`notebooks/01_direction.ipynb`](notebooks/01_direction.ipynb) (the direction
question, end to end) and
[`notebooks/02_volatility_and_planning.ipynb`](notebooks/02_volatility_and_planning.ipynb)
(the answerable questions). Everything below is the short version, with pointers
into the code.

---

## What goes in

Public data only — free, reproducible, committed to this repo so it all runs
offline: daily closes from Yahoo Finance (2015 onward), the dollar index and
VIX, interest rates and CPI from FRED, weekly futures positioning from the
CFTC, and 99 FOMC statements. Downloads: [`scripts/01`](scripts/01_download_data.py),
[`08`](scripts/08_download_macro.py), [`09`](scripts/09_download_cot.py),
[`11`](scripts/11_download_fomc.py); offline validation of every stored file:
[`scripts/16_check_data.py`](scripts/16_check_data.py).

Three guards run through everything:

1. **One-day lag on every market feature.** Found the hard way: Yahoo stamps FX
   closes earlier in the day than the US index closes, so a naive same-date join
   leaks the future. The check that caught it
   ([notebook 1 §1](notebooks/01_direction.ipynb)):

   ```python
   print(f"corr(dxy_t, eurusd_t)   = {eur.corr(dxy):+.2f}")        # -0.09
   print(f"corr(dxy_t, eurusd_t+1) = {eur.shift(-1).corr(dxy):+.2f}")  # -0.87  <- the leak
   ```

   Before the fix one model showed 70% daily accuracy. After it: baseline.

2. **Frequency matched to the question** — daily bars predict the next day,
   weekly the next week, monthly the next month.

3. **Walk-forward throughout** — every model refit as of each prediction date,
   never scored on data it trained on
   ([`src/regime_lab/data/splits.py`](src/regime_lab/data/splits.py)).

---

## The chart that predicts the whole study

Autocorrelation answers "does knowing the past k days tell you anything about
today?" — asked twice on the same prices: of returns (direction) and of
absolute returns (size).

![Autocorrelation of returns vs absolute returns](outputs/case_study/figures/autocorrelation.png)

Size has memory at every lag on every pair; direction has essentially none.
That asymmetry is the study's outcome in one picture: direction models are
fighting noise, volatility models are not.

---

## Outcome 1 — the usable one: fix next month's rate from today's close

Budgets and contracts need one rate agreed in advance, and what they implicitly
forecast is next month's **average** — its own target with its own correct
baseline. Three estimates compared at each month-end, 104 months per pair
([`scripts/14_period_average.py`](scripts/14_period_average.py), notebook 2 §5):

| Pair | Last month's average | **Today's rate carried forward** | AR model | Improvement |
|---|---|---|---|---|
| EURUSD | 1.21% | **1.01%** | 1.03% | −16% |
| USDINR | 0.90% | 0.66% | **0.65%** | −28% |
| USDJPY | 1.51% | **1.13%** | 1.14% | −25% |

*Error as % of the realised monthly average; lower is better.*

![Planning-rate error by estimator](outputs/case_study/figures/planning_rate_error.png)

The AR model roughly ties the simple carry-forward everywhere — the gain comes
from choosing the right baseline, not from modelling. **The operational change:
reset the planning rate at month-end from the latest close instead of the
trailing average.** No model to deploy, no data to buy, and it held on all
three pairs.

---

## Outcome 2 — the forecastable one: size, not direction

Ask how big next week's moves will be and the problem becomes tractable. A
standard HAR model (yesterday's, last week's and last month's volatility as
inputs) against carrying today's volatility forward, walk-forward on
non-overlapping windows, 426 weeks per pair
([`scripts/10_turbulence_har.py`](scripts/10_turbulence_har.py), notebook 2 §4):

| Pair | Carry vol forward | **HAR** | HAR + ML |
|---|---|---|---|
| EURUSD | 0.840 | **0.360** | 0.416 |
| USDINR | 1.721 | **0.841** | 0.990 |
| USDJPY | 1.130 | **0.550** | 0.662 |

*QLIKE, the standard volatility loss; lower is better.*

![EURUSD next-week volatility, forecast vs realised](outputs/case_study/figures/har_forecast_eurusd.png)

HAR roughly halves the error on every pair. Note the third column: **adding
machine learning on top made forecasts worse in five of six cells** (monthly
horizon included). Complexity has to earn its place; here it did not, and that
negative is reported as part of the result.

What you'd use it with: a volatility forecast doesn't say which way the rate
moves — it says how much cushion to leave. Wider spread, earlier hedge, smaller
position in a week forecast rough; the reverse in a calm one.

---

## Outcome 3 — the dead ends

Fifteen models were graded on direction — trend and carry rules, institutional
and retail playbooks, logistic regression, gradient boosting, random forest, a
hidden Markov model, an ensemble (notebook 1 §4–7, full bench in
[`scripts/07_build_dashboard.py`](scripts/07_build_dashboard.py)). Against the
laziest baseline available — *the next period repeats the last one* — on the
held-out final period:

| Pair | Best edge vs floor | 95% interval | Reading |
|---|---|---|---|
| EURUSD | +6.4pp | [−1.8, +15.5] | crosses zero |
| USDJPY | +1.6pp | [−4.9, +8.8] | crosses zero |
| USDINR | +1.4pp | [−5.3, +8.7] | crosses zero |

One result did survive, and it's about confidence rather than accuracy:

![Calibration on the held-out block](outputs/case_study/figures/calibration_block6.png)

Logistic regression's stated probabilities track reality; gradient boosting
says "10%" on days that come up 37% of the time and "90%" on days that come up
56% — far more confident than its hit rate justifies. An accuracy leaderboard
would never show this; scoring probability quality did.

The second dead end was tested rather than assumed: all 99 FOMC statements
scored hawkish / neutral / dovish with a transparent dictionary method
([`scripts/12_stance_score.py`](scripts/12_stance_score.py)) —

![FOMC stance scores 2015–2026](outputs/case_study/figures/fomc_stance_timeline.png)

— which validates as a *measurement* (the 2022–23 hiking cycle reads hawkish,
the 2020 emergency cuts dovish) but adds nothing as a *predictor*: with and
without the stance features on 402 identical weeks, both probability-score
differences span zero ([`scripts/13_stance_ablation.py`](scripts/13_stance_ablation.py)).
Rejected, and kept as a documented negative.

---

## Method, in four lines

- **A named floor for every target** — persistence and majority-class for
  direction, carry-forward vol for turbulence, carry-forward rate for averages.
  Half the findings here are baseline corrections.
- **Intervals on every headline claim** — block-bootstrap CIs with sample sizes
  stated ([`src/regime_lab/eval.py`](src/regime_lab/eval.py)).
- **Confidence scored, not assumed** — probability quality graded alongside
  accuracy; overconfident models recalibrated on held-out data or labelled raw.
- **Fully reproducible** — data, pipeline, models and every chart regenerate
  from the numbered scripts; the notebooks and tables agree because they share
  [`src/regime_lab`](src/regime_lab).

Known limits: three pairs only; volatility proxied from daily closes (no
intraday data); CPI features EURUSD-only; no rupee futures contract; FOMC
covers the dollar leg only; no transaction costs modelled. Detail and full
tables: [`outputs/case_study/FINDINGS.md`](outputs/case_study/FINDINGS.md).

## Run it

```bash
uv sync
uv run pytest                              # 83 tests
uv run python scripts/16_check_data.py     # validates every stored data file
uv run jupyter lab notebooks/
```

Nothing here is investment advice.

## License

MIT. See `LICENSE`.
