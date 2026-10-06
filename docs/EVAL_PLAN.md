# Evaluation plan: Legacy planner vs QStats planner

Committed on 2026-10-06, after the development runs on the dev SKUs and **before** the comparison
was run on the demo SKUs. The plan below is not edited after that run; corrections go to the
change log at the end, with the numbers before and after.

## Question

Run the same simulated business twice from the same moment: once with the Legacy planning process,
once with QStats. Same products, same real demand pattern, same supplier luck. Does QStats deliver
the same service with less inventory, or more service for the same inventory?

## Set-up

| Item | Value |
|---|---|
| Business | Fictional U.S. importer (two DCs + Amazon FBA); demand patterns from UCI Online Retail II; everything else simulated (`config/demo.yaml`, business seed 42) |
| Populations | `demo`: 200 SKUs (reported). `dev`: 120 different SKUs (every tuning decision). Both chosen on the first 52 weeks only |
| Shared history | Weeks 1–12 warm-up (ample stock, not scored). Weeks 13–52: the Legacy planner runs the business and creates the censored history |
| Fork | Start of week 53. World L keeps Legacy; World Q hands the same state to QStats. Every variant forks from the same week-52 state |
| Scoring window | Fork (2024-12-02) + longest quoted lead time (68 days) to the last complete week of data: 2025-02-08 to 2025-12-07, 303 days (shifted calendar) |
| Replicate worlds | Supply seeds 42 (reference), 7, 2026: supplier lead times, delays, disruptions, FBA behaviour and feed gaps change; products and demand do not |
| Headline SKUs | Demo SKUs without a simulated promotion or liquidation (168 of 200). The 32 event SKUs are reported separately, because the size of a simulated uplift is a dial the designer controls |

### Environment parameters and why

| Parameter | Value | Why |
|---|---|---|
| Quoted lead time | Median of the supplier's true distribution | The quote is honest on average; QStats's lead-time model can only gain from the spread, not from a planted bias |
| Lead-time spread | Log-sd 0.08–0.22 per PO, plus a shared supplier-week shock 0.04–0.12 | Orders from one supplier in one week share their luck |
| Delays | 4–12% of POs, mean 8–20 extra days | Ocean freight tails |
| Disruptions | 0.4 per supplier on average, 3-week windows, +15–30 days | Supplier-wide events |
| Cross-DC fulfilment | Allowed, $1.10 extra per unit | What an order-management system does when one DC is out |
| Amazon channel | Small-basket customers of FBA-enabled SKUs (median line ≤ 12 units), 45% of them | Wholesale-sized buyers do not shop on Amazon |
| Wholesale lots | Lines > 10× the SKU's 95th-percentile line and ≥ 1,000 units are left out of demand (4.2% of units) | A handful of lines would otherwise decide every fill-rate metric |
| Promotions | 12% of SKUs, 1–3 events, uplift lognormal (mean log 0.55, sd 0.30), unknown to both planners; calendar announced 6 weeks ahead to both | Exercises event handling; excluded from the headline |

## Policies

**Legacy** (`simulation/policies/legacy.py`): SES on weekly sales (alpha 0.2, tuned on dev), closed
weeks skipped, sales taken as demand; quoted lead time as certain; safety stock = 30 days of
forecast demand; weekly review; order up to lead-time + review + safety demand; same MOQ and
case-pack rounding as QStats; FBA by days of cover (pick + transit + review + 14 days trigger,
+30 days target).

**QStats** (`replenishment/policy.py`): stockout reconstruction (local_profile, chosen on dev);
champion forecast per segment by rolling-origin cumulative error (25 candidates, seasonal prior on
or off); Kaplan–Meier supplier lead times with open POs censored; demand over lead time + review as
an exact mixture of lead-time and forecast-error distributions; order-up-to = the product's
service-level quantile; RESERVED FBA units counted at 0.5.

**Variants run from the same fork**

| Family | Variants |
|---|---|
| Legacy frontier | 15, 30 (reference), 45, 60, 75, 90, 120 days of safety stock |
| Legacy + seasonal prior | 30 days, SES on seasonally adjusted sales, re-seasonalised over the lead time |
| QStats frontier | Product targets (reference: A 97%, B 95%, C 90%), and uniform 70, 80, 90, 95, 98% |
| Ablation (2×2×2) | Stockout reconstruction on/off × seasonal prior on/off × probabilistic safety stock and lead times on/off (off = quoted lead time, 30 days of cover). All on = the QStats reference |
| Sensitivity worlds | `null`: lead time always equals the quote, no events. `optimistic_quotes`: quotes at the 25th percentile of the true distribution. Reference seed only |

## Metrics

Per SKU over the scoring window, summed before ratios, so every world is scored on identical
SKU-days.

- **Fill rate** = units sold / true demand (hidden baseline).
- **In-stock rate** = 1 − stocked-out channel-days / active channel-days (direct channel = both
  DCs pooled; Amazon separately).
- **Average inventory** = mean daily on-hand units plus units in transit to Amazon, at landed cost.
- **Working capital** = average inventory + supplier shipments in transit (owned from shipment).
- Lost units and lost contribution; inventory turns (annualised COGS / average inventory);
  excess at the end (stock beyond 26 weeks of the last 13 weeks' true demand); purchase value of
  POs placed from the fork; ending position (on hand + on order) so no policy looks lean by
  starving the end; cross-DC shipments.
- **Forecast WAPE and bias**: one-week-ahead network forecast vs true weekly demand.

## Primary metric and how it is read

**Inventory QStats needs to deliver the Legacy process's fill rate**, read by linear interpolation
along the QStats frontier at the fill rate of Legacy-30, relative to Legacy-30's inventory. No
extrapolation: if the fill rate lies outside the QStats frontier the result is "out of range".

- Reported for seed 42 with a 90% paired bootstrap interval over SKUs (1,000 resamples; every
  resample uses the same SKUs in every variant). Seeds 7 and 2026 are reported as point estimates.
  Their spread is supply luck, not the uncertainty of the estimate, and is labelled that way.
- If the 90% interval includes zero, the README says the saving at today's service level is not
  distinguishable from zero.

**Secondary**: inventory the Legacy process would need to reach QStats's fill rate (interpolated
on the Legacy frontier, no extrapolation), relative to QStats's inventory.

**Ablation**: for every arm, the inventory the Legacy frontier needs at that arm's fill rate,
relative to the arm's own inventory (positive = less stock for the same service). This controls
for the operating point, which differs between arms.

**Calibration**: realised coverage of the P50/P80/P90/P95 of demand over lead time + review,
using the lead time the SKU's order would have had that week, by segment and by season.

**Forecast accuracy**: no claim of better accuracy unless QStats's WAPE is lower in all three
replicate worlds.

## The one-shot demo run

```
python scripts/evaluate_policies.py --population demo
python scripts/evaluate_policies.py --population demo --seeds 42 --scenario null
python scripts/evaluate_policies.py --population demo --seeds 42 --scenario optimistic_quotes
```

Run once, after this file is committed. Any later change to planner code that would change these
numbers is logged below with the numbers before and after.

## Decisions made on the dev SKUs

| Decision | Choice | Evidence (dev) |
|---|---|---|
| Legacy alpha | 0.2 | One-week MAE on visible sales: 0.1 → 41.5, 0.2 → 39.9, 0.3 → 40.3 |
| Reconstruction method | local_profile | Two-sided episode MAE 142.9 (censored_gamma 150.2, pre/post velocity 145.9, none 251.1) |
| Error model | Additive errors scaled by max(26-week, all-history mean) | Ratio errors gave 80–400× tails for intermittent SKUs and 19× the Legacy inventory |
| Seasonal prior shrink | Log space, weight n/(n+10) | Linear shrinkage flattened the Christmas shape from ~30× to ~7× |
| De-trending the prior | Not for the Christmas group | Their January–August build-up is season, not trend |
| Frontier ranges | Added Legacy 75/90/120 days, QStats 70/80% | The first run's frontiers did not overlap |
| Legacy FBA rule | Counts pick + transit + review time | In 75% of FBA stockout days the DCs had stock: the first version was mis-specified, not a legacy weakness |
| Cross-DC fulfilment | Allowed | WEST fill 70% vs EAST 84% with no cross-shipping, which no order system would accept |

The same rules, re-applied after the third and fourth corrections to the dev SKUs as redrawn by the
selection fix (change log). Latest: legacy α stays 0.2 (one-week MAE 38.5 vs 40.0 at 0.1 and 39.0 at
0.3); the reconstruction method is censored_gamma (two-sided episode MAE 119.0 vs local_profile
124.0, pre/post velocity 130.7, model expectation 161.3, none 223.5).

## Development results (first dev population, headline subset, 101 SKUs)

These informed the decisions above. The dev SKUs were redrawn in the third run and the comparison
was not rerun on them, so this section is a record of what was known before the demo run.

| | Legacy-30 | QStats |
|---|---|---|
| Fill rate (seed 42) | 87.0% | 92.6% |
| Average inventory | $131.8k | $195.0k |
| One-week WAPE | 0.592 | 0.587 |
| Forecast bias | −14.0% | −7.7% |

Primary metric on dev: 7.8% less inventory at Legacy-30's fill rate (seed 42; 90% interval −3.1%
to 13.5%); seeds 7 and 2026: −1.7% and 5.2%. Secondary on dev: Legacy needs 43% more inventory to
reach QStats's fill rate on seed 42, about the same on seed 7 (−0.5%). The dev evidence for a
saving at today's service level is weak, and the size of the high-service advantage depends on
supply luck. Calibration on dev: P80 covered 77.5%, P90 86.1%, P95 90.0% (narrow, worse in the
peak season and for new products). Null world (dev, seed 42): 9.4% saving, interval 1.8% to 14.2%.

## Test results (demo SKUs; after the fourth correction, 2026-10-06; see the change log)

Headline subset: 168 demo SKUs without simulated events (the same 168 in every world). Scoring
window 15 Feb – 7 Dec 2025, 296 days (see the change log: it now starts when QStats's first orders
can arrive). Reference world (supply seed 42) unless stated. Sign convention for the two matched
metrics: positive = QStats holds less inventory.

| | Legacy-30 | QStats |
|---|---|---|
| Fill rate | 84.0% | 92.3% |
| In-stock rate | 87.0% | 90.1% |
| Average inventory | $247.1k | $439.3k |
| Lost contribution | $146.1k | $57.9k |
| Carrying cost (24% a year, 296 days) | $48.1k | $85.5k |
| Cross-DC shipping | $28.1k | $16.4k |
| Inventory turns | 3.55 | 2.19 |
| Excess at the end | $41.3k | $155.5k |
| Ending position (on hand + on order) | $405.7k | $636.1k |
| One-week WAPE | 0.616 | 0.645 |
| Forecast bias | −20.5% | −7.3% |

**Primary metric: not distinguishable from zero.** To deliver Legacy-30's fill rate QStats needs
4.7% *more* inventory (90% interval: from 4.8% less to 16.6% more). In 26.9% of the 1,000 resamples
Legacy-30's fill rate fell below QStats's lowest setting (70% target), outside the frontier; the
interval is computed over the 73.1% in range. In 6.9% of all resamples that lowest setting also
held no more inventory than Legacy-30, so the excluded resamples lean toward QStats. Replicate
worlds: 2.2% more (seed 7), 2.6% more (seed 2026). At the service level the current process
delivers, QStats does not save inventory.

**Secondary metric: out of range in the reference world.** QStats's fill rate (92.3%) is above the
highest point of the Legacy frontier (120 days of safety stock: 92.1% fill with $559.7k, 27.4% more
inventory than QStats's $439.3k), and the plan does not extrapolate. 63.6% of the resamples are out
of range for the same reason. Replicate worlds: the Legacy rule needs 4.8% more (seed 7) and 16.6%
more (seed 2026) inventory than QStats to reach QStats's fill.

**Money over the 296 scored days:** QStats recovered $88.1k of contribution and saved $11.7k of
cross-DC shipping, at $37.4k of extra carrying cost: about +$62k against Legacy-30, while ending
with $114.1k more stock beyond 26 weeks of demand and $230.4k more on hand and on order. Legacy-120,
at nearly the same fill (92.1%), nets about +$33k against Legacy-30 on the same terms, so about half
of QStats's figure is the higher service level itself.

**Service targets vs fill.** The targets (A 97%, B 95%, C 90%) are cycle service levels: the
chance that what is ordered now covers demand until the next order arrives. Realised unit fill
(92.3%) is lower because fill is a different measure, because the intervals run narrow (below), and
because the 168 SKUs include new products, products that die, and Amazon channels that never sold
(below).

**Class targets vs one target.** In all three worlds QStats at a uniform 95% beat the class targets
it uses: more fill with less stock (seed 42: 93.0% with $427.5k vs 92.3% with $439.3k; seed 7:
92.0% / $375.0k vs 91.3% / $408.9k; seed 2026: 92.5% / $380.4k vs 91.4% / $399.6k). The class
targets were fixed before the run and are kept; setting targets by margin is on the roadmap.

**Ablation (inventory the Legacy frontier needs at the arm's fill rate / arm's inventory − 1;
seeds 42 / 7 / 2026; — = outside the Legacy frontier). The third switch changes safety stock, lead
times (quoted vs Kaplan–Meier), the FBA rule and the weight on RESERVED Amazon units (1.0 vs 0.5)
together:**

| Arm (reconstruction · seasonal prior · safety stock, lead times and FBA rule) | Efficiency vs Legacy frontier |
|---|---|
| Legacy + seasonal prior | +19.2% / +20.4% / +24.3% |
| — · — · 30-day (13 candidates, raw sales) | — / — / — (below Legacy-15's fill, with more inventory than it, in every world) |
| — · — · probabilistic | −44.9% / −48.5% / −46.6% |
| — · prior · 30-day | — / — / −14.0% (below Legacy-15's fill, with more inventory than it, in seeds 42 and 7) |
| — · prior · probabilistic | −39.5% / −45.9% / −38.2% |
| reconstruction · — · 30-day | −5.3% / −6.9% / −1.2% |
| reconstruction · — · probabilistic | +15.8% / −13.0% / −10.9% |
| reconstruction · prior · 30-day | +6.2% / +2.4% / +13.5% |
| reconstruction · prior · probabilistic (QStats) | — / +4.8% / +16.6% (above Legacy-120's fill in seed 42) |
| QStats at 90% for every SKU | +27.2% / +22.2% / +8.3% |
| QStats at 95% for every SKU | — / +30.2% / — (above Legacy-120's fill in seeds 42 and 2026) |

Inside QStats no ingredient helps alone: reconstruction alone costs efficiency, probabilistic
safety stock learned from raw, censored sales scores −44.9% to −48.5%, the forecaster alone sits
below the whole Legacy frontier in every world, and the prior alone does in two of three (−14.0% in
the third). With reconstruction, the prior helps in all three worlds and probabilistic safety stock
in one. **Legacy + prior is the strongest single change** (+19.2% to +24.3%); full QStats can be
read in two worlds and Legacy + prior beats it in both (20.4% vs 4.8%; 24.3% vs 16.6%).

**Forecast accuracy: worse, and no improvement claimed.** WAPE 0.645 vs 0.616, 0.653 vs 0.621,
0.651 vs 0.622 (QStats vs Legacy, three worlds). Bias falls from −20.5 / −19.7 / −19.5% to
−7.3 / −6.8 / −7.0%. The live plan's out-of-sample challenger (forecasting methodology) suggests
the SKU-level model override fits noise.

**Calibration (QStats, all 200 SKUs, 6,387 overlapping SKU-weeks, no interval):** the P50 covered
55.0%, P80 75.8%, P90 84.1%, P95 89.2%. Too narrow, most for new products (P90 57.9%, P95 67.2%;
458 SKU-weeks) and in the peak season (P95 85.0%). REGULAR SKUs, 469 SKU-weeks, are close to
nominal (P90 90.4%, P95 94.7%).

**Amazon channels that never sold.** 17 of 72 FBA-enabled SKUs in World Q (18 in World L) never
sold on Amazon after the fork; 15 of them (16) had Amazon demand, 7,884 units in World Q (7,957 in
World L) that went unserved. With no stock there is no sale, so neither planner saw that demand: a
channel that never sells looks exactly like a channel with no demand.

**Event SKUs (32: 24 with promotions, 8 with a liquidation; reported separately):** QStats 91.8%
fill with $48.2k inventory; Legacy-30 89.0% with $36.3k.

**Sensitivity worlds (same 168 SKUs):** with lead times always equal to the quote and no events,
QStats saves 5.3% at Legacy-30's fill rate (interval from 6.0% more to 12.6% less; 42.6% of
resamples out of range), and the Legacy rule needs 39.5% more inventory to reach QStats's fill
(interval +12.1% to +59.3%; 9.8% out of range). QStats's edge in this replay does not come from
modelling lead-time uncertainty: it was larger when lead times were certain. With optimistic
quotes (25th percentile) Legacy-30's fill drops to 80.9%, below QStats's lowest setting (85.2%), and
QStats reaches 92.7%, above Legacy-120's 92.0%: both matched metrics are out of range (98.4% and
90.4% of resamples).

### Known limitations of this evaluation

- The pre-registration is a local git commit (`7a98f5e`) with no external timestamp; it shows the
  order of work only to someone who trusts the history.
- One demand path. Replicate worlds vary supply luck only.
- The bootstrap resamples SKUs and ignores clustering by supplier (ten suppliers share weekly
  shocks and disruptions), so its intervals are likely too narrow.
- Each frontier has six or seven points and is interpolated linearly; an effect of a few percent is
  within interpolation error.
- Fill rate is a ratio of sums, so large SKUs dominate it; no per-SKU distribution is reported.
- Error quantiles come from the champion's errors on the windows used to select it (a winner's
  curse), and windows more than 10% reconstructed are not scored, which leaves out busy stockout
  periods. Both likely contribute to the narrow intervals.
- QStats now runs near the top of the Legacy frontier, so the secondary metric falls out of range
  in the reference world. Legacy settings above 120 days were not run, and adding them after seeing
  the results would be a change the plan does not allow; the dominance over Legacy-120 is reported
  instead.
- The reported numbers follow four rounds of corrections. The third changed the SKU sample (a
  selection bug) and, by re-applying the pre-set development rule, the reconstruction method; the
  change log attributes the movement between runs.
- The replay approves everything the plan sizes: every order and Amazon send, including the lines
  the live plan holds for a person (low margin, low forecast confidence, sends marked for review),
  and no container top-ups. It measures the sizing rules, not a planner's judgement on top.

## Change log

- 2026-10-06: plan committed (`7a98f5e`); demo run once the same day.
- 2026-10-06, correction: `weighted_quantiles` interpolated between support points, which
  understates quantiles of a mixture with discrete atoms (a test with a two-point lead time gave
  a P90 of 400 instead of 600). Replaced with the inverted-CDF quantile and the comparison rerun,
  nothing else changed. Primary metric −2.61% → −2.62% (interval −14.3%/+6.5% → −14.4%/+6.5%;
  seeds −1.03% → −1.03%, +0.89% → +0.88%). Secondary 5.1% → 5.0% (seeds 12.9% → 13.5%, 9.4% →
  9.7%). QStats average inventory $353.5k → $353.9k, ending position $462.1k → $458.4k. Ablation
  rows moved by at most 3.1 points (QStats 95%, seed 2026: 26.5% → 29.6%). The conclusions do not
  change. A second fix (backtest padding when history is shorter than 24 weeks) cannot affect
  these runs, which always have at least 52 weeks.
- 2026-10-06, correction after an independent read of the results: (1) the sensitivity world with
  certain lead times had been scored on all 200 SKUs (it has no events) while the headline uses 168;
  it is now scored on the same 168. Saving at Legacy-30's fill 7.7% (interval 0.1% to 13.3%) →
  8.7% (1.4% to 12.6%). (2) Bootstrap resamples outside the frontier are now counted and reported
  (10.5% for the primary metric). (3) The primary interval in this section read −14.3%; the value
  after the first correction is −14.4%. (4) The ablation table now shows all arms; the earlier
  version left out the arms that add nothing. (5) Planner code was moved so it no longer imports the
  simulation (location constants and `PlannerView` to `domain/`); the rerun reproduced every base
  result exactly (largest difference 0.0).
- 2026-10-06, third correction, after a second independent audit of code, numbers and wording (98
  confirmed findings; wording fixes are not listed here). Corrections that change results:
  1. **SKU selection.** Eligibility and profile metrics were computed on series zero-filled up to
     each SKU's last sale in the full two-year file, and "still selling at the end of the selection
     year" used that last sale: year-two information leaked into a year-one selection. Both now use
     data up to the end of the selection year only. This redrew 94 of the 200 demo SKUs and 67 of
     the 120 dev SKUs.
  2. **Seasonal prior panel.** The same leak decided which products counted as selling through the
     first year: 1,508 → 1,490 products (1,419 general, 71 Christmas-type).
  3. **Keywords** were matched as substrings (STAR matched START, TREE matched STREET); now whole
     words. (This covered the seasonal keywords only; the category keywords followed in the fourth
     correction.)
  4. **Censored-gamma reconstruction** ignored its configured dispersion floor (0.3) and clipped at
     1.0.
  5. **Champions and error tables were fitted on the two-sided reconstruction,** which uses data
     after each backtest origin; the backtest now uses the reconstruction as it stood on the day.
     Forecasts still use the two-sided one.
  6. **Hysteresis** judged a SKU-level pick on segment scores; it now compares on the SKU's own
     windows when both models are scored there.
  7. **Seasonal naive** was dropped whenever the prior was off (ablation arms); it is now a
     candidate in every arm (13 without the prior).
  8. **The 30-day ablation arms** sized cycle stock as the mean of the error mixture instead of the
     point forecast over the quoted lead time + review, which is what the legacy rule uses.
  9. **Order split:** the rounding remainder went to EAST_DC whenever its fractional share was at
     least 0.5; it now goes to the preferred DC.
  10. **Scoring window** started at the fork + the longest quote, but QStats's first orders are
      placed a week into the fork. It now starts at fork + review + longest quote: 8 Feb → 15 Feb
      2025, 303 → 296 days.
  11. **Calibration** used the lead time of the day before the order was placed.
  12. **Inventory-feed gaps** were 5–14 days long, not the documented 5–15.
  13. **Future trading days** assumed a 23 Dec – 3 Jan shutdown; the shifted history shows
      27/28 Dec – 6/7 Jan.
  14. The `PlannerView` kept the engine's raw purchase-order records (true arrival days) in a private
      attribute. No planner read it; it now holds only what a buyer would see.

  Re-applying the development rule to the redrawn dev SKUs kept legacy α = 0.2 and switched the
  reconstruction method to censored_gamma (see the decisions above).

  Effect on the reference world (seed 42), one step at a time. The two middle rows were run only to
  attribute the change and are not reported results:

  | Step | Fill, Legacy-30 / QStats | Primary (90% interval; out of range) | Secondary (90% interval; out of range) | WAPE, QStats / Legacy |
  |---|---|---|---|---|
  | Before | 82.0% / 88.7% | −2.6% (−14.4% to +6.5%; 10.5%) | +5.0% (−14.9% to +25.0%; 2.1%) | 0.633 / 0.623 |
  | Code and environment fixes (same SKUs, local_profile) | 81.9% / 89.4% | −1.1% (−15.8% to +4.8%; 28.8%) | +11.5% (−10.5% to +25.8%; 17.2%) | 0.653 / 0.618 |
  | + redrawn SKUs | 84.0% / 92.3% | −6.4% (−20.3% to +4.6%; 21.4%) | out of range (60.5%) | 0.638 / 0.616 |
  | + censored_gamma (the run reported after the third correction) | 84.0% / 92.3% | −2.3% (−14.2% to +6.3%; 37.8%) | out of range (66.0%) | 0.640 / 0.616 |

  At today's service level the conclusion holds at every step: no saving distinguishable from zero.
  The secondary metric went out of range with the new SKU sample, on which QStats runs above the top
  of the Legacy frontier. Replicate worlds, primary: −1.0% / +0.9% → +0.6% / −3.2% (seeds 7 /
  2026); secondary: +13.5% / +9.7% → +7.1% / +28.2%. Null world: saving 8.7% (1.4% to 12.6%) → 5.6%
  (−5.7% to +12.8%); Legacy extra +19.9% → +37.1%. Calibration barely moved (P90 83.9% → 83.8%, P95
  89.1% → 89.1%). The full run was repeated after adding the out-of-range bookkeeping below and
  reproduced every number exactly.

  Corrections that leave the comparison unchanged: the ETS challenger is now out of sample (the
  in-sample version reported the opposite, 0.394 vs 0.306; out of sample 0.504 vs 0.670); the
  bootstrap records which side of the frontier an out-of-range resample falls on; weekly forecast
  bands use single-week errors; recommendation text for SKUs with no forecast demand;
  CRITICAL_STOCKOUT value booked as a loss; containers pack only BUY lines and top up no
  low-confidence SKU; the Scenario simulator calls the plan's own code; one excess definition; the
  app's WAPE skips mostly reconstructed weeks; the benchmark's episode end day and weekly table; the
  week-by-week chart limited to headline SKUs.
- 2026-10-06, fourth correction, after a third independent review (four lenses, each finding
  checked by a second reviewer: 38 confirmed). A clean clone of the third correction's commit had
  first reproduced every artefact byte for byte. Corrections that change results:
  1. **SKU override and hysteresis compared models on different windows.** Each model was averaged
     over its own scored windows, so seasonal naive (no forecast where last year's week was closed)
     could win on windows its rival was never scored on. Both now compare two models only on the
     windows where both were scored, with at least three of them.
  2. **Category keywords** (which also decide a SKU's supplier) were still matched as substrings: TIN
     matched GREETING, SIGN matched DESIGN, so greeting cards sat in Kitchen & Dining. Now whole
     words, plurals allowed. 13 of the 200 demo SKUs changed category and 12 changed supplier.
     Re-applying the development rules to the changed dev world kept legacy α = 0.2 and
     censored_gamma (see the decisions above).

  Effect on the reference world (seed 42): fill Legacy-30 / QStats 84.0% / 92.3% → 84.0% / 92.3%
  (QStats inventory $435.8k → $439.3k); primary −2.3% (−14.2% to +6.3%; 37.8% out of range) →
  −4.7% (−16.6% to +4.8%; 26.9% out of range); secondary out of range in both (66.0% → 63.6% of
  resamples). Replicate worlds, primary +0.6% / −3.2% → −2.2% / −2.6%; secondary +7.1% / +28.2% →
  +4.8% / +16.6%. Legacy + prior against full QStats: in both worlds where both can be read, Legacy +
  prior now wins (+20.4% vs +4.8%, +24.3% vs +16.6%; before, one of two). WAPE 0.640 → 0.645
  (Legacy 0.616). Null world: saving 5.6% → 5.3%, Legacy extra +37.1% → +39.5%. Calibration P90
  83.8% → 84.1%, P95 89.1% → 89.2%. The conclusions at today's service level do not change; the
  replicate worlds now lean slightly more against QStats.

  Corrections that leave the comparison unchanged: the plan's money and service figures are now
  timing-aware (open POs on their expected dates, unmet demand lost): a purchase is credited only
  with demand it serves after it lands, contribution at risk and projected fill cover the next 13
  weeks, and cycle service is labelled as the backorder view it is; the Scenario simulator grades
  the legacy rule's orders on the same yardstick as the plan's and applies the lead-time knob to it;
  weekly forecast bands are zero in closed weeks; container top-ups are counted in the week's
  commitment; CI lint no longer depends on a local data/ folder; QSTATS_CONFIG resolves like every
  other path; `make clean` and `make plan` keep the Docker pipeline marker in step; a run without
  the reference seed is refused instead of overwriting its artefacts; the replay's automatic
  approval of review lines is documented; and wording across the README, the docs and the app.

