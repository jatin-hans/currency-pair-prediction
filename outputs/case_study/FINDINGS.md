# When is FX direction predictable? A calibrated, abstaining evaluation

**Findings document — every number regenerates from a named script; figures in `figures/`, tables in `tables/`.**
Data: daily closes 2015–2026 (yfinance), three currency pairs — EURUSD, USDJPY, USDINR. Code and methods: this repository.

## The question

Can a model predict whether a currency pair will move up, down, or sideways over the next five trading days — and, more importantly, can it *tell you when it doesn't know*? This is a case study in honest evaluation: standard models, pre-registered comparisons, confidence intervals on every published number, and a selective predictor that is allowed to abstain.

## Task setup, in one paragraph

Target: the direction of the 5-day-ahead log return — UP, FLAT, or DOWN, where FLAT means the move is smaller than 0.25 × the trailing 20-day volatility × √5 (so "sideways" scales with how noisy the market currently is). That dead zone captures 20–26% of days per pair, and the split is roughly 40/20/40 (`tables/`). Evaluation uses ordered time-series cross-validation: six expanding train windows with non-overlapping test windows from 2017 to 2026, with 5 rows purged at every boundary so no training label can peek at test prices. Blocks 1–5 are exploratory; **Block 6 (2025-01 → 2026-04) is the held-out confirmation set** for the two comparisons pre-registered before any model was fit. One-day-ahead prediction was excluded up front: at that horizon the dead zone is smaller than the vendor's price-snapping noise, so a "direction" label would measure the data feed, not the market.

## The leak we found and fixed (2026-08-03)

During the revision audit we discovered that the dollar-index and VIX series were stamped one trading day ahead of the FX closes (Yahoo stamps FX daily bars earlier than the US index closes): for EURUSD, corr(dxy_return_t, fx_return_{t+1}) was **−0.87** versus −0.10 contemporaneous — "today's" dollar move sat inside the return being predicted. Every result in this document was regenerated after lagging both series one trading day (`src/regime_lab/data/features.py`, `scripts/07_build_dashboard.py`). The effect was large: a dashboard daily-window cell that read 70% accuracy collapsed to roughly the majority-class baseline, and the pre-fix headline below — "USDJPY +9.4pp over the floor, CI excludes zero" — became +1.6pp, CI [−4.9, +8.8]. The pre-fix numbers survive in this repository's history for the record.

## Findings

### 1. After the fix: no accuracy edge anywhere; one small probability-quality edge survives

Headline metric: **balanced accuracy** — the average of the model's accuracy on UP days, FLAT days, and DOWN days, so random guessing scores 33% regardless of class mix. Logistic regression on ten standard features (returns, realized volatility at three horizons, dollar-index co-movement, VIX, calendar — the eleven-column feature table minus the raw price) vs the naive floor (persistence forecast), Block 6 confirmation set, with a block-bootstrap CI on the *gap*:

| Pair | Logistic | Floor | Gap | 95% CI on gap | Verdict |
|---|---|---|---|---|---|
| EURUSD | 40.8% | 34.4% | +6.4pp | [−1.8, +15.5] | positive but CI includes zero |
| USDJPY | 32.8% | 31.2% | +1.6pp | [−4.9, +8.8] | no detectable skill |
| USDINR | 33.2% | 31.7% | +1.4pp | [−5.3, +8.7] | no detectable skill |

The probability-based score (paired Brier difference, which uses each day's full forecast rather than one right/wrong bit and therefore has tighter intervals) finds the one surviving positive: EURUSD −0.036 [−0.073, −0.001] excludes zero — the simple model's stated probabilities are measurably better-behaved than the floor's on that pair. (This B1−B0 comparison was **not pre-registered** — the registered primaries are P1 and P2 below — and its CI's upper bound sits at −0.001; treat it as provisional, per this document's own multiple-comparison rule.) USDJPY +0.003 [−0.008, +0.013] and USDINR +0.008 [−0.007, +0.023] do not. *(Provenance: `scripts/03_run_baselines.py` → `tables/paired_diffs_baselines.csv`, `tables/metrics_baselines.csv`.)*

### 2. The flexible model was confidently wrong — the case for scoring probabilities

Gradient boosting posts respectable *accuracy* but its probability estimates are so overconfident that its Brier score is **worse than the naive floor on every pair and every scope** — unchanged by the leak fix (Block 6: EURUSD +0.146 [+0.071, +0.212], USDINR +0.191 [+0.111, +0.267], USDJPY +0.169 [+0.095, +0.249]; blocks 1–5 all similar, every CI excludes zero). The reliability diagram (`figures/r4_reliability.png`) shows the mechanism: its high-confidence calls happen far less often than stated, while logistic regression's stated probabilities roughly match observed frequencies. **A leaderboard scored on hit-rate would have picked the wrong model.** *(Provenance: `scripts/03_run_baselines.py`, `scripts/06_export_case_study.py`.)*

### 3. Pre-registered result P1: conditioning on volatility state added nothing measurable

The hypothesis that fitting separate models per market condition (calm / normal / turbulent, defined by a causal trailing-volatility tercile) beats one pooled model was tested as registered — Brier(condition-aware GBT) − Brier(pooled GBT), Block 6 (post-fix): EURUSD +0.051 [−0.026, +0.141], USDINR −0.002 [−0.085, +0.074], USDJPY +0.047 [−0.035, +0.133]. **No CI excludes zero. The refutation is the result.** (The supplementary accuracy view agrees after the fix: −0.8pp / −2.5pp / −3.4pp, none excluding zero — the pre-fix "conditioning helps EURUSD, hurts USDJPY" pattern did not survive.) The sparsity guard (fall back to a pooled model when a condition has < 250 training rows) fired exactly where designed — on the earliest, smallest training windows (`tables/n1_fallbacks.csv`). *(Provenance: `scripts/04_run_condition_aware.py` → `tables/p1_primary.csv`.)*

### 4. Where skill lives (exploratory) — post-fix answer: nowhere reliably

Before the leak fix this section reported per-state pockets of skill. After the fix the per-state view (all blocks pooled, `figures/r2_state_heatmap.png`) shows the simple model's probability score is *worse* than the floor in most states — significantly so in calm markets on all three pairs and in every USDINR state (e.g. EURUSD calm +0.026*, USDINR turbulent +0.040*, USDJPY calm +0.031*; * = 95% CI excludes zero) — a hangover of the early exploratory blocks, since the Block-6-only comparison in §1 is better. The honest reading: pooled over ten years there is no market state where these features reliably beat persistence; the pre-fix "skill concentrates in turbulent states" story was partly the leak talking. *(Provenance: `scripts/04_run_condition_aware.py` → `tables/state_skill_diffs.csv`.)*

### 5. Pre-registered result P2: a model that knows when not to answer

A conformal wrapper (split conformal, thresholds set per volatility state on a rolling calibration window) turns the logistic model into a selective predictor: answer only when exactly one direction clears the confidence threshold. At the pre-committed 80% target on Block 6 (post-fix):

| Pair | Empirical coverage [CI] | Within ±5pp? | Abstains | When it answers |
|---|---|---|---|---|
| EURUSD | 0.708 [0.630, 0.783] | no — under-covers by 9pp | 63% of days | n = 119, 57% correct (chance 33%) |
| USDJPY | 0.792 [0.724, 0.857] | yes | 98% of days | n = 6, no meaningful sample |
| USDINR | 0.960 [0.928, 0.988] | no — over-covers | 99% of days | n = 4, no meaningful sample |

Post-fix, the honest summary is harsher than before: on the two pairs with no demonstrated skill the wrapper behaves correctly by **refusing to answer virtually every day** — that is what "knowing what it doesn't know" looks like — but on EURUSD, the one pair where it answers often (37% of days, 57% correct), it *misses* its 80% coverage target by 9 points. Coverage here is *empirical*, never guaranteed: FX days are not exchangeable, so conformal theory's finite-sample guarantee does not transfer — and Block 6 shows exactly that failure mode rather than hiding it. Per-state coverage swings on small turbulent subsets and is shown, not hidden (`tables/conformal_state_coverage.csv`). *(Provenance: `scripts/05_run_conformal.py` → `tables/conformal_coverage.csv`, `tables/conformal_sweep.csv`.)*

## When your labels disagree (the investigation that predates the results)

Before any model was fit, the project's two candidate "market stress" labels — a volatility tercile and a drawdown flag — were found to disagree wildly on USDINR (correlation 0.118 vs ~0.3–0.4 on G10 pairs). The investigation (April 2026) traced it to a real property of the pair: USDINR grinds *upward* through its volatile episodes, so drawdown-based stress flags miss them. The drawdown label was later deleted outright when review found its threshold used the full sample — a look-ahead. Finding and killing your own leakage is part of the method, and the paper trail exists.

## How this was built

Failing-first tests for every defect fix and module (106 tests); leakage invariants enforced in code (ordered splits, purged boundaries, trailing-only features — with a test that mutates the future and asserts the past doesn't change); pre-registered primaries fixed at R0 before any model ran; approval-gated phases with a decision ledger; every table and figure regenerable from `scripts/01` → `scripts/06`; fixed seeds; no hyperparameter search — all budgets fixed and documented in `src/regime_lab/models.py`.

## Honest accounting

- **Comparisons run: 84** (36 in R1, 21 in R2, 27 in R3), of which **6 were pre-registered** (P1, P2 × three pairs). Everything else is labeled exploratory where it appears. With 84 cells, a handful of spurious "significant" exploratory results are expected; that is why the headline claims are only the pre-registered ones.
- **The headline metric was switched to balanced accuracy after the first results were reviewed** — a presentation choice for readability, disclosed here because metric switches after seeing results are exactly the kind of freedom pre-registration exists to constrain. The two registered comparisons remain scored precisely as registered (Brier for P1, coverage for P2), and both metric views are published for every comparison.
- **Effect sizes are small.** A 3.6-point Brier improvement on one pair and 57%-correct selective answers are real but modest; transaction costs would dominate any attempt to trade this, which is why there is no backtest and no P&L anywhere in this study.
- **Block 6 is one period.** One confirmation window is evidence, not proof; a rolling re-confirmation as new data accrues would strengthen it.
- **All results were regenerated on 2026-08-03 after fixing the DXY/VIX timestamp leak** (see the section at the top). The pre-registered comparisons were re-scored under identical, pre-committed protocols — nothing else changed. The pre-fix numbers survive in this file's history; the leak, its magnitude, and its effect on each headline are disclosed inline rather than silently overwritten.

## Why isn't the HMM better? (added 2026-07-31)

A Gaussian hidden Markov model on returns was added to the dashboard tier on owner request, with the expectation it might dominate. Diagnostic (per pair, 3-state HMM on daily log returns — now exported to `tables/hmm_states.csv`, script `scripts/15_hmm_states.py`): two of three states are nearly identical in mean and volatility (EURUSD: 46.5 vs 45.9 bp daily vol, ~49% occupancy each), and their state-conditional direction frequencies sit within ~1pp of the marginal base rate — no usable directional information. Only the rare high-vol state (0.9–3.2% of days) tilts (EURUSD stormy: P(UP)=0.46 vs 0.39 marginal; USDJPY stormy: P(DOWN)=0.52 vs 0.37), too rare and too retrospective to win. The HMM learns volatility clustering — its legitimate, famous use — not direction, and it sees strictly less information than the feature models (returns only). This reproduces a 30-year-old literature result: Engel (1994) — Markov-switching can't beat a random walk out-of-sample; Dacco & Satchell (1999) — why regime models forecast poorly; modern consensus — regimes are volatility states, useful for risk control, not direction.

## What the published literature says (added 2026-07-31, from a two-agent survey)

- **Short-horizon direction in major pairs is essentially unpredictable from public data.** Meese & Rogoff (1983) still stands (Rossi 2013 survey; Kılıç 2025, Fed, for ML specifically). The only credible daily-horizon predictor is proprietary dealer order flow (Evans & Lyons 2002; but Sager & Taylor 2008: commercially available flow gives no edge).
- **What has replicated evidence works elsewhere:** carry (interest differentials; crash-prone, Sharpe roughly quartered post-2008), 1–12-month momentum/trend (the only price-only method; genuine in the 1970s–80s, gone in liquid majors since the mid-1990s — Neely, Weller & Ulrich 2009; survives at multi-month horizons, Moskowitz-Ooi-Pedersen 2012, Hurst-Ooi-Pedersen 2017), value/PPP (3–5-year half-life, Rogoff 1996). All are portfolio risk premia, not point forecasts.
- **Meta-facts:** currency anomalies lose ~66% of returns post-publication (Bartram, Djuranovik & Garratt); in the M6 competition only 8.6% of teams beat the naive benchmark with significance (Makridakis et al. 2024); ML papers claiming 55–90% daily FX accuracy collapse to ~50–55% under leakage/cost scrutiny (Dautel et al. 2020); time-series foundation models score ~51% directional (Rahimikia 2026).
- **Implication for this study's conclusion:** our post-fix result — no accuracy edge over persistence anywhere, one small probability-quality edge on EURUSD, and a self-caught look-ahead bug that had been manufacturing a large fake daily edge — is what the literature predicts for public price data at these horizons, including the part where impressive numbers turn out to be leakage. It is a *replication of the consensus*, not a failed hunt. The honest route forward is a different question (turbulence; period averages), not a fancier direction model.

## Dashboard surface note (added 2026-07-31; accuracy-upgrade revision same day)

The live dashboard (`scripts/07_build_dashboard.py`) is frequency-matched: daily bars predict the next day, weekly bars (Friday closes) the next week, monthly bars the next month, with all feature windows scaled in bars. Nine tiers at first (now fifteen — see the revision sections): floor, trend following, carry rule, logistic, gradient boosting (isotonic-calibrated on a time-ordered split), random forest (calibrated), naive Bayes, HMM, and an ensemble averaging the four learned tiers (two of them isotonic-calibrated). Its recent scorecards are tiny samples by construction and are labeled as such; this frozen study remains the evidentiary core.

**Accuracy-upgrade levers applied (owner-approved 2026-07-31):**
- *Lever 1 (partial):* calibration of the overconfident tree models + the ensemble tier. The abstention machinery was deliberately left untouched per owner ruling (no increase in no-answer days).
- *Lever 2:* multi-horizon momentum features (1/3/6/12 months in bars) and cross-pair return/momentum features. Event-calendar dummies deferred — no reliable free machine-readable feed found.
- *Lever 3:* FRED macro via `scripts/08_download_macro.py` — interest-rate differentials for all pairs (US 3-month Treasury, ECB deposit rate, Japan 3-month interbank with a 60-day publication-lag shift, India 10-year yield as a documented proxy with the same shift) feeding all learners plus a carry rule tier; CPI value-gap feature for EURUSD only (no live Japan/India CPI on FRED — disclosed, not patched). COT positioning and a wider currency panel remain on the roadmap.
- *Lever 4:* a turbulence sub-model per pair × frequency (logistic on volatility features, predicting whether the next bar's move exceeds the trailing median) reported alongside direction — the built-in demonstration that volatility is predictable where direction barely is.

## Scorecard post-mortems (added 2026-07-31)

- **Carry rule on USDINR (6/26 weekly, 3/12 monthly): the instructive failure kept on display.** Plain version: holding rupees is like renting out a house in a slowly declining neighborhood — the rent (India's ~3.2pp higher interest) is great, but the property value (the exchange rate) drifts down yearly with India's higher inflation; carry investors profit from rent-minus-drift combined, while our dashboard grades only the price move and never counts the rent. Technical version: carry's replicated evidence is for *excess returns* (spot move + interest accrual); our target is spot direction only, so on an EM pair with a persistent inflation differential the constant "rupee strengthens" call is systematically wrong-way. Textbook Balassa/inflation-drift mechanics, shown rather than hidden. (Flipping the rule post-hoc would score at most ~17/26 — FLAT weeks cap it well below the naive "20/26" a two-class intuition suggests — and would be data snooping anyway, so we don't.) Two further disclosures: India has no free live short rate, so the rule compares a US 3-month rate with an Indian **10-year** yield, and that gap has favoured the rupee every week since 2015 — making the rule a constant, not a signal.
- **EURUSD's rough recent stretch is a turning-point story, told honestly.** Post-fix, the main-six learned models score 9–10 of 26 recent weeks (random forest 13) — below the 14/26 an always-UP caller would score in that window; the positioning tier and the HMM each reached 15/26. The cause is a real regime turn: EURUSD's 3/6-month momentum is still negative and the trend signal still reads "downtrend" while the pair has rallied since April 2026 — models trained on the decline kept calling DOWN through the turn. Monthly cells are n=12 anecdotes. (An earlier draft claimed the ensemble was "the board's best cell at 15/26 vs the floor's 10" — that was wrong even against the pre-fix payload and is retracted.)
- **Evaluation protocol, stated precisely:** every graded call at every frequency is one-step-ahead on actual observed data. "Recent" calls retrain at each prediction date on all data resolved by that date; "context" runs use one frozen fit but still read actual bars at each prediction date. Predictions are never fed back as inputs; no multi-step forecast chains exist anywhere in the pipeline.

## Trader-playbook benchmark tiers (added 2026-08-02, two-agent research)

Owner request: research how institutions and individuals actually trade these pairs and add their playbooks as rule-based benchmarks. Two research agents surveyed academic papers, fund prospectuses, published rulebooks, and broker/regulator data. Four tiers were selected for the strongest documentation and full daily-close implementability, joining the dashboard (13 tiers at that point; 15 after the smart-money tier and the conditional-momentum refinement):

- **Institutional — multi-speed time-series momentum** (`tsmom`): majority sign-vote of the trailing 1/3/12-month returns, the AQR-style spec (Moskowitz-Ooi-Pedersen 2012 JFE; Hurst-Ooi-Pedersen 2017, positive in every decade since 1880; stated in AQR's fund prospectus). Trend exposure explains most of real currency managers' returns (Pojarliev-Levich).
- **Institutional — Turtle Donchian breakout** (`turtle`): the one complete institutional rulebook ever published verbatim — long on a 20-bar closing high, short on a 20-bar low, flat on a 10-bar reverse extreme (System 1 stance, symmetric).
- **Retail — RSI(14) 70/30 fade** (`rsi`): the most-taught retail rule *and* the one matching measured retail behavior — broker-flow studies show retail trades contrarian (negative-feedback), which this encodes. Academic evidence for it in FX majors: weak to negative (Hsu-Taylor-Wang 2016).
- **Retail — Bollinger (20, 2σ) band fade** (`bollinger`): the second contrarian retail archetype.

Honest scope limits, stated on the page: we replicate *entry logic* only and grade direction — institutional edge is substantially position sizing (vol targeting, 2N stops) which direction-grading cannot capture; retail's dominant loss driver is the disposition effect (average winner ≈ half the average loser, FXCM data), not the entry signal. Context stats from the research: 74–89% of retail FX/CFD accounts lose (ESMA); retail USDINR futures were effectively ended by RBI's May-2024 hedging-only mandate (volumes −87%). Not implemented (data-impossible or intraday): order-flow strategies, London-fix/session trades, stop-cluster hunting; deferred: FOMC-day rule (needs the meeting calendar), month-end rebalancing rule.

## Real trades and real positions (added 2026-08-02, two-agent research)

Owner asked for actual trades by profitable investors, as a timeline with their reasoning, feeding the rule engine. Findings:

- **A verified famous-trades timeline** now lives on the page: Soros's Plaza Accord dollar short (1985), Druckenmiller's post-Wall Deutsche Mark long (1989), the 1992 sterling break (the "200% of net worth" exchange, per the Lost Tree transcript), the 1994 $600M yen-short *failure*, the Abenomics USDJPY short (Bessent/Soros, ~$1bn in 3 months; Bass parallel), Druckenmiller's 2022–23 dollar short on trend maturity, and the FX Concepts $12bn→bankruptcy anti-rule. Each card carries the trader's quoted reasoning, the distilled rule, and whether our engine runs a version (trend-riding → momentum tiers; policy divergence → rate-gap features/carry; sizing and thesis-stops → out of scope for a direction-only engine, disclosed).
- **Negative results, stated plainly:** no named, verified profitable USDINR trade exists in public (rupee speculation runs through anonymous offshore NDFs); and there is no public trade-by-trade dataset of elite FX traders at all — famous trades surface only via books, speeches, and position leaks. Retail copy-trading records (Myfxbook/eToro) are public but are "dumb money" per the academic literature.
- **The one public window on institutional positions is the CFTC Traders-in-Financial-Futures report** — actual weekly hedge-fund net positions in euro and yen futures. `scripts/09_download_cot.py` fetches 2015→present (604 weeks), stamps rows with publication dates (+3 days, Tuesday-positions-published-Friday) for causality, and feeds: two features (positioning z-score vs its 2-year range — the practitioner-documented extremes signal, cf. Danske's 16th/84th percentile rule — and 4-week change) plus a **smart-money tier** that simply follows the funds' net position. No rupee contract exists; USDINR has no COT tier and the dashboard's model list adapts per pair.
- **Evidence honesty:** the academic literature (Klitgaard-Weir NY Fed 2004; Hossfeld-Röthig 2016; RBNZ 2018) finds fund positioning *tracks* weekly FX moves rather than leading them; only positioning extremes and carry-crowding-as-crash-risk have replicated support. The smart-money tier is therefore framed as a live test of a popular belief, not a validated edge. Early scorecards: 15/26 on weekly EURUSD, 10/26 on weekly USDJPY — appropriately mixed.

## What I'd do differently

Calibrate gradient boosting (isotonic/Platt on a validation fold) before conformalizing, rather than discarding it for overconfidence; try adaptive conformal for the drifting USDINR case; pre-register the per-state analysis next time (it turned out to be the most interesting exploratory result); and extend Block 6's end date on a schedule, as a logged config change.

---

*Through-line for the portfolio: the same design principle as the Momos routing case study — confidence-based routing there, abstention here. Systems should know when not to answer.*
