# Forecasting methodology

Forecasting is an input to purchasing, so it is judged on what a purchase rests on: total demand
over the protection interval (supplier lead time + review period), not one week ahead.

## Models

| What is forecast | Model | Fitted on | Compared against |
|---|---|---|---|
| Weekly demand per SKU, 1–26 weeks | Champion per segment among 25 candidates (below), with a SKU-level override | Reconstructed demand up to the plan date | Every other candidate, by rolling origin; a per-SKU ETS, out of sample |
| Demand on stockout days | Censored likelihood: local level × weekday × season, with sold-out days treated as a lower bound (gamma, EM; chosen on dev) | Clean and sold-out days around each episode | No adjustment and three alternatives, against hidden true demand |
| Demand over lead time + review | Mixture of lead-time and forecast-error distributions | Backtest errors and supplier receipts to date | Realised demand (calibration) |

Every candidate works on the seasonally adjusted rate per trading day,

    y_t = x_t / (e_t · s_t)          x units, e trading days in the week, s seasonal prior (or 1)

and forecasts a level with an optional damped trend:

    SES             ℓ_t = α y_t + (1 − α) ℓ_{t−1}                         α ∈ {0.1, 0.2, 0.3}
    Damped Holt     ℓ_t = α y_t + (1 − α)(ℓ_{t−1} + φ b_{t−1})
                    b_t = β(ℓ_t − ℓ_{t−1}) + (1 − β) φ b_{t−1}             (α, β, φ) ∈ {(.2,.05,.9), (.3,.1,.85)}
                    ŷ_{t+h} = ℓ_t + (φ + … + φ^h) b_t
    Croston         z, p updated only when y_t > 0;  ŷ = z / p             α = 0.15
    SBA             ŷ = (1 − α/2) z / p
    TSB             π_t = β 1[y_t > 0] + (1 − β) π_{t−1};  ŷ = π_t z_t       (α, β) = (0.15, 0.1)
    Naive, moving average (4, 8, 13 weeks), seasonal naive (52 weeks)

    units ahead:    x̂_{t+h} = ŷ_{t+h} · e_{t+h} · s_{t+h}

Each of the twelve non-seasonal models runs with and without the seasonal prior (24), plus
seasonal naive. With the prior switched off (an ablation arm) the twelve run once, plus seasonal
naive: 13 candidates. Weeks before launch, with no trading days, or inside a promotion or liquidation
do not update a model's state. All recursions run vectorised over SKUs, so one pass over the
history gives the forecast from every origin.

### Why there is no per-SKU Holt-Winters

At the moment QStats takes over there is one year of history, and two by the end of the data. A
52-week seasonal index per SKU would have no degrees of freedom left. Seasonality comes instead
from a **prior** built once, before the replay, from 1,490 other products that were selling within
four weeks of both the start and the end of the first year (judged on first-year data only):
monthly rate per trading day, each product's January–August trend removed (except for
Christmas-type products, whose January–August build-up is their season), divided by its own mean,
median across products, in two groups chosen by whole-word description keywords. Shrinkage is in
log space with weight n/(n+10): 0.99 for the general group (1,419 products), 0.88 for the
Christmas group (71). Daily factors interpolate between mid-month points. The prior is ablated in
the comparison, and "Legacy + the same prior" is a row of it.

## Rolling-origin validation

Never a random split. From every week w, each model forecasts the following weeks with data up to
w only, and the forecast is compared with what happened. Kept per origin:

- cumulative forecast and actual over h = 1, 3, 6, 8, 11, 14, 17 and 24 weeks;
- the one-week-ahead forecast, for WAPE, MAE, RMSE and bias;
- in the live plan only, the single week 1, 3, 6, 11, 17 and 24 weeks ahead, for the weekly bands.

A window is scored only if it is complete and **at most 10% of its actual demand was
reconstructed**, so the reconstruction method does not grade the forecast. MAPE is not used: many
weeks have zero or tiny demand.

## Champion selection

    score(model, segment) = mean over SKUs and scored origins in the last 52 weeks of
                            |cumulative forecast − cumulative actual| / (mean weekly units × h)

with h the SKU's protection interval (lead time P50 + review: 6.6 to 10.9 weeks in the live plan)
rounded to the nearest of 6, 8, 11 or 14 weeks. Chosen **per segment**, because one SKU's own
record is short: over 52 weeks it holds three to eight non-overlapping windows of that length, few
to choose among 25 candidates.

- Parsimony: a simpler model within 2% of the best wins.
- Hysteresis: an incumbent is replaced only if the new pick is 5% better, on the SKU's own windows
  when both models are scored there, otherwise on the segment's.
- SKU override: a SKU keeps its own best model only if it beats the segment champion by 15% on at
  least three non-overlapping windows.
- Re-selection every four weeks in the replay; the live plan selects once.

The override was meant as an exception. In the live plan 138 of the 200 SKUs use it (35 on three
windows, 46 on four, 21 on five, 33 on six, one on seven, two on eight), and the out-of-sample
check below suggests those picks fit noise.

Segments (as of the plan date, `demand/segmentation.py`), first rule that matches: NEW_PRODUCT
(< 26 weeks since launch), INTERMITTENT (average demand interval ≥ 1.32 weeks), SEASONAL
(Christmas-type), TRENDING (|relative change over 26 seasonally adjusted weeks| ≥ 0.35), VOLATILE
(CV² of non-zero weeks ≥ 0.49), REGULAR. Because INTERMITTENT comes first, most Christmas products,
which sell on few weeks, land there; the SEASONAL segment is small.

**Challenger (as of the plan date, not used to plan).** statsmodels ETS(A, Ad, N) with smoothing,
trend and damping fitted by maximum likelihood (`forecasting/challengers.py`). Both sides are out
of sample: the champion is the one each SKU had 26 weeks before the plan date, chosen on data up
to then, and ETS is refitted at every fourth origin of those 26 weeks with data up to that origin.
A window counts only when both have a forecast, scored with the selection's scaled error. Over 702
windows on 182 SKUs ETS scored 0.504 against the champions' 0.670:

| Segment | SKUs | ETS | Champion |
|---|---|---|---|
| INTERMITTENT | 97 | 0.620 | 0.961 |
| TRENDING | 58 | 0.375 | 0.290 |
| VOLATILE | 18 | 0.348 | 0.295 |
| REGULAR | 7 | 0.236 | 0.253 |
| SEASONAL | 2 | 0.529 | 1.233 |

The champions hold up only for trending and volatile SKUs. An earlier version of this check scored
both on the windows the champions had been chosen on, and reported the opposite (0.394 vs 0.306);
out of sample the comparison reverses. Testing ETS as a candidate and tightening the SKU override
are on the roadmap; a global gradient-boosting challenger is too.

## Uncertainty

For the champion of each SKU, every scored origin gives a scaled error

    e = (actual − forecast) / (level × h)

where level is the larger of the SKU's mean weekly units over its last 26 usable weeks and over its
whole history, as of the origin. Errors are pooled by segment and horizon bucket (1, 2–4, 5–8,
9–13, 14–20, 21–26 weeks) and kept as 200 quantiles. Demand over h weeks is then
max(0, forecast + level × h × e).

The weekly bands shown with a plan use a second table of the same shape: errors of the single week
h ahead, e = (actual − forecast) / level. Week-to-week noise dominates this demand: pooled over
segments, the P10–P90 spread of those errors grows by about 16% from one week ahead to 21–26 weeks
ahead, so the bands widen only a little (and narrow where the forecast falls toward zero).

Two earlier versions were rejected on the dev SKUs: ratio errors (actual / forecast) produced
80–400× tails for intermittent SKUs whose forecast was near zero before a large order, and a 26-week
level alone did the same for SKUs dormant for months.

**Calibration is measured, not assumed.** In the controlled replay QStats recorded its quantiles at
every weekly plan; the realised value uses the lead time an order placed that week would have had.
Over 6,385 SKU-weeks (all 200 SKUs, overlapping windows, no interval computed) the P80 held 75.3% of
outcomes, the P90 83.8% and the P95 89.1%. The intervals are too narrow, most for new products
(P90 57.2%, P95 66.6%) and in the September–December season (P95 85.2%); REGULAR SKUs (479
SKU-weeks) are close to nominal (P90 89.8%, P95 94.6%). Two likely causes, both by construction: the error
quantiles come from the champion's errors on the same windows used to select it (a winner's curse),
and windows more than 10% reconstructed are not scored, which leaves out busy stockout periods.
Recalibration is the first roadmap item.

## Confidence

HIGH / MEDIUM / LOW from a documented score (`replenishment/recommendations.py`): history length,
segment (NEW_PRODUCT, INTERMITTENT, VOLATILE lower it), share of the last eight weeks that was
reconstructed, and the champion's backtest error. A purchase with LOW confidence is routed to
REVIEW_FORECAST instead of BUY.

## What the evidence says about accuracy

In the replay, QStats's one-week WAPE was 0.640, 0.652 and 0.644 in the three worlds against the
legacy process's 0.616, 0.621 and 0.622: QStats was less accurate week to week in all three, and no
accuracy gain is claimed. Forecast bias fell from about −20% to about −7% in all three, which is
what stockout correction is for. The out-of-sample challenger above points at the SKU-level
override as one likely cause of the accuracy gap.
