# Key insights: method, dead ends, pitch shortlist

Source material for the two app tabs ("Methodology & Validation", "Things We Tried but Didn't Work Out") and the 3-minute pitch. Written for people outside the ML side.

All numbers are averages over **28 simulated 3-day outages** (July to September; see M1) unless marked otherwise. Raw numbers: `experiments/results.csv`; history: `experiments/LOG.md`. Each "Visual" names a ready-made Plotly figure in `pipeline/figures.py` or a section of `notebooks/01_key_visuals.ipynb`.

---

## 1. Methodology & Validation

**M1. We judge the model on 28 simulated outages, not on one week.**
We hide 3 days of park T at a time and predict them without looking, 28 times. The official validation week (4–9 Sep) alone is misleading. There the baseline's park total was off by only +0.1%, but only because T1 (+19%) and T3 (-8%) cancelled out. Over the 28 outages the same baseline was off by 11% on average.
As a check we repeated the test with an outage starting every day (83 overlapping outages). The result held: the park energy error fell from 10.8% (baseline) to 8.3% (final), and the final model was better in two out of three outages.
*Visual:* `fig_distribution` (the error of every simulated outage).

**M2. The model in one sentence.**
T wind = a + b × (park H wind) + c × (park O wind), fitted separately for each turbine.
- The offset a and the H factor b are adjusted for each of 8 wind directions.
- The inputs are 1-hour averages of the neighbours.
- Every coefficient can be printed and explained; there is no black box.

**M3. Short gusts are noise, trends are predictable.**
Gusts at a neighbour do not reach T at the same moment. Matching the neighbours' 10-minute values to T's agrees less (correlation 0.80) than matching 3-hour averages (0.89). So we feed the model 1-hour averages of the neighbours, which lowers the wind error.
*Visual:* `fig_smoothing`, notebook section 5.

**M4. Wind direction changes how much wind T gets.**
How much of park H's wind T receives depends on the direction:
- wind from the north / north-east: T gets 84% of H's wind;
- wind from the east / south-west: T gets 93%.

That is terrain and park layout. Accounting for it cuts the per-turbine energy error from 13.8% to 11.8%.
*Visual:* `fig_direction`.

**M5. Average the power, not the wind (our biggest single gain).**
Between 3 and 11 m/s, where 97% of the energy is produced, the power curve bends upward. A gust above the predicted wind therefore adds more power than an equally large lull below it removes.
- **Before:** feeding the single best-guess wind speed into the power curve made the baseline underestimate park energy by 8.5% on average.
- **After:** we average the power over the range of plausible wind speeds, and that bias drops to +1.4%.
- **On its own,** this change lowers the park energy error from 11.0% to 9.0%.

*Visual:* `fig_power_evidence`, `fig_bias`.

**M6. The park energy error is simply the 3-day average wind error.**
Over the 28 outages, how far the park energy is off almost exactly follows how far the average predicted wind is off over the 3 days (correlation -0.92). Three reasons:
- **Errors are short-lived.** A wind error at one moment is largely gone 3 hours later (its correlation with itself falls from 0.35 after 1 hour to 0.11 after 3 hours and about 0 after 6 hours). Three days therefore behave like about 24 independent pieces. Their average does not reach zero; it leaves a random offset worth about ±11% of energy per outage.
- **The three turbines are wrong together** (correlation 0.64–0.84), so adding them up does not cancel the error.
- **The edges don't help.** The error just before and just after an outage tells us nothing about the error inside it (correlation about 0). Correcting with it made results worse (park error 7.8% → 10.9% at half strength).

Consequence: the remaining error is mostly weather luck that the neighbour data cannot explain. We do not hide it; we report it as a range (M7).
*Visual:* `fig_bias_noise` (the bias went from -8.5% to +1.4%; the ±11% noise stayed).

**M7. Uncertainty is shown, and it is calibrated.**
- **Every 10-minute step** gets a 95% band. Its width depends on the wind level, because errors grow with wind speed. In the simulated outages the true wind fell inside the band 94.8% of the time (target 95%).
- **The lost energy** gets a 95% range built from the energy errors of the 28 simulated outages: -15% to +30% around the estimate. The range is lopsided because calm periods produce large relative errors.
- **Is that range honest?** For each of the 83 daily outages we built the range from the other outages only. The true energy fell inside it 93% of the time, close to the 95% target.

*Visual:* `fig_calibration`, `fig_example`.

**M8. Results: baseline vs final model** (28 simulated outages)

| Metric (judged) | Baseline | Final | Change |
|---|---|---|---|
| Wind speed RMSE | 1.261 m/s | 1.200 m/s | -5% |
| Wind speed MAE | 0.941 m/s | 0.889 m/s | -5% |
| Power RMSE | 737 kW | 682 kW | -7% |
| Power MAE | 478 kW | 457 kW | -4% |
| Park energy error (mean absolute, per outage) | 11.0% | 8.5% | -2.5 points |
| Park energy bias (mean signed) | -8.5% | +1.4% | bias removed |
| 95% band coverage | 94.6% | 94.8% | on target |

Lost energy for the judged outage, 1–3 Sep: **322 MWh (95% range 247–378 MWh)**.
*Visual:* `fig_scorecard`, `fig_winrate` (the final model beats the baseline in 85–90% of the outages on wind and power RMSE).

**M9. Built-in safety rules.**
- The model never sees park T during the outage, and the code checks this on every run.
- If an extra input (wind direction, for example) is missing at some step, that step falls back to the plain baseline formula, so the prediction never has gaps.

**M10. The judged outage has favourable weather.**
1–3 Sep is 89% south-west wind and windier than average: 6.8 m/s at H against about 5.3 m/s in a typical simulated outage. South-west is the direction where the neighbours track T best (correlation 0.83, against 0.72 from the north-east).
- **Windy outages are easier.** The relative energy error falls with wind speed: 10.7% in calm, 8.0% in medium and 6.3% in windy outages.
- **Outages like 1–3 Sep are the easiest of all.** In the 13 simulated outages that resemble the judged one (mostly south-west, windy), the final model was off by 3.2% on average, the baseline by 7.8%. That is a small sample, so we keep the wider range from M7 for the official answer.

---

## 2. Things we tried but didn't work out

The lesson in one line: most "obvious" additions add numbers to the model without adding information. Two things mattered: giving the model the right inputs (hour averages plus wind direction) and converting wind to power correctly (M5).

| Idea | Why it looked promising | What happened (park energy error, baseline 11.0%) | Our reading |
|---|---|---|---|
| Weight recent days more | The relation drifts from July to September (offset a: 1.23 → 0.27) | 11.0–11.6%, no better or worse | The drift is not local; less data only adds noise |
| Stretch the prediction to T's natural variability (variance-ratio method, an industry standard) | Plain regression flattens extremes, which biases energy | Bias almost gone (-2%), but wind error up from 1.26 to 1.33 m/s and validation week off by +13% | It adds noise at every step; averaging the power (M5) removes the bias without that cost |
| Curved relations (log, square root, piecewise lines) | Calm and strong wind might behave differently | 10.7–11.9% | A straight line is good enough |
| Time of day | T gets relatively more wind in the afternoon (ratio 0.85–1.00) | 11.4% | No gain beyond what wind level and direction already carry |
| Temperature, air pressure, turbulence | Atmospheric stability changes wind profiles | 11.0–11.6% | No information beyond the neighbours' wind speed |
| Black box (gradient boosting, all inputs) | Captures any pattern automatically | 9.7% (final transparent model: 8.5%); band covers only 89% | **The transparent model was better and is explainable** |
| All 7 neighbour turbines separately (instead of park averages) | More detail | 10.5% | Small gain, more coefficients |
| Time-shifted neighbour values (±20 min) | Wind travels between parks | Same accuracy as 1-hour averages | The H factor got split over 5 near-identical columns, which nobody can interpret; replaced by averages |
| Neighbour values 30 min later | H trails T by about 30 min | Correlation 0.795 → 0.806 | Negligible; the 1-hour average covers it |
| Correct with the error just before and after the outage | Tracks the local weather situation | Park error 7.8% → 10.9% (half strength), 17.2% (full) | Errors are gone after 3 hours (M6); the edges say nothing about the middle |
| Out-of-sample or "conformal" uncertainty bands | Often better calibrated than training errors | Coverage 94.6% → 94.7–94.8% | Training errors were already calibrated; making the width depend on wind speed was what mattered |
| A separate park-O factor per wind direction | Lean on whichever park is upwind | 8.6–8.9% (final model: 8.5%) | Direction is already captured by the offset and H factor per direction; extra terms add numbers, not information |
| Learn mostly from hours with the same wind direction as the outage | The weather situation might matter more than the date | 8.5% (tie) | The model is already linear in the right inputs; re-weighting the same data changes nothing |
| Automatic choice of reference turbines | Bonus criterion; fewer inputs | 8.3% vs 8.3% (tie on the daily test) | Same accuracy with only 2–3 references: O1 alone carries most of the information, plus one or two H turbines. **A good story for the bonus, but no accuracy gain** |
| Small neural network (same inputs) | Flexible fit | 9.1% | Worse than the transparent formula, like the gradient boosting |
| Wind direction, for the **park total** | Strong effect per turbine (13.8% → 11.8%) | Park total 11.0% → 10.9% | Direction fixes differences *between* turbines, which cancel when the park is summed anyway. Kept because it improves wind and power accuracy |

*Visual:* `fig_leaderboard` (all about 40 configurations), `fig_blackbox`.

---

## 3. Pitch shortlist (3 minutes: pick 3–4)

Ranked by how much they help the story. ★ = recommended.

| # | Message (one line on a slide) | Proof point | Visual |
|---|---|---|---|
| ★1 | "We tested on 28 simulated outages, not one lucky week." | The validation week looked perfect (+0.1%) but hid ±19% per turbine | `fig_distribution` |
| ★2 | "Average the power, not the wind." | Bias -8.5% → +1.4%; the largest single gain (11.0% → 9.0%) | `fig_power_evidence` |
| ★3 | "What remains is weather luck, so we show a range." | Park error = 3-day average wind error; ±11% per outage. Hidden window: 322 MWh (247–378) | `fig_bias_noise` |
| ★4 | "The transparent formula beat the black box." | 8.5% vs 9.7%; one line of maths, every number explainable | `fig_blackbox` |
| 5 | "Wind direction decides how much wind reaches T." | 84–93% of H's wind depending on direction | `fig_direction` |
| 6 | "Gusts are noise; we use hourly trends." | Correlation 0.80 (10 min) vs 0.89 (3 h) | `fig_smoothing` |
| 7 | "Most extras added nothing." | Time weighting, time of day, temperature, pressure, curves | Table in section 2 |
| 8 | "The judged outage is our best-understood weather." | 89% south-west wind, windy; in similar simulated outages the error was about 3% (small sample) | Direction rose (to build) |
| 9 | "The model picks its own neighbours." | O1 plus one or two H turbines give the same accuracy as all seven | Bar chart of how often each reference is chosen (to build) |

Suggested 3-minute flow:
1. The problem and the insurance question (20 s).
2. The model in one sentence, M2 (30 s).
3. ★1 → ★2 → ★3, ending on the kWh range (90 s).
4. ★4 as the explainability argument (20 s).
5. A pointer to the live app (20 s).
