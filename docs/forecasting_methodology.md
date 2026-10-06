# Forecasting methodology

Forecasting is an input to purchasing, so it is judged on what a purchase rests on: total demand
over the protection interval (supplier lead time + review period), not one week ahead.

## Models

| What is forecast | Model | Fitted on | Compared against |
|---|---|---|---|
| Weekly demand per SKU, 1–26 weeks | Champion per segment among 25 candidates (below) | Reconstructed demand up to the plan date | Every other candidate, by rolling origin |
| Demand on stockout days | Local level × weekday × seasonal profile (chosen on dev) | Clean days around each episode | No adjustment and three alternatives, against hidden true demand |
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
seasonal naive. Weeks before launch, with no trading days, or inside a promotion or liquidation
do not update a model's state. All recursions run vectorised over SKUs, so one pass over the
history gives the forecast from every origin.

### Why there is no per-SKU Holt-Winters

At the moment QStats takes over there is one year of history, and two by the end of the data. A
52-week seasonal index per SKU would have no degrees of freedom left. Seasonality comes instead
from a **prior** built once, before the replay, from 1,508 other products that sold through the
whole first year: monthly rate per trading day, each product's January–August trend removed
(except for Christmas-type products, whose January–August build-up is their season), divided by
its own mean, median across products, in two groups chosen by description keyword. Shrinkage is
in log space with weight n/(n+10): 0.99 for the general group (1,508 products), 0.88 for the
Christmas group (74). Daily factors interpolate between mid-month points. The prior is ablated in
the comparison, and "Legacy + the same prior" is a row of it.

## Rolling-origin validation

Never a random split. From every week w, each model forecasts the following weeks with data up to
w only, and the forecast is compared with what happened. Kept per origin:

- cumulative forecast and actual over h = 1, 3, 6, 8, 11, 14, 17 and 24 weeks;
- the one-week-ahead forecast, for WAPE, MAE, RMSE and bias.

A window is scored only if it is complete and **at most 10% of its actual demand was
reconstructed**, so the reconstruction method does not grade the forecast. MAPE is not used: many
weeks have zero or tiny demand.

## Champion selection

    score(model, segment) = mean over SKUs and scored origins in the last 52 weeks of
                            |cumulative forecast − cumulative actual| / (mean weekly units × h)

with h the SKU's protection interval rounded to 6, 8, 11 or 14 weeks. Chosen **per segment**: with
one or two years of weekly data and a 10–15 week horizon, one SKU has two or three independent
windows, too few to choose among 25 candidates without fitting noise.

- Parsimony: a simpler model within 2% of the best wins.
- Hysteresis: an incumbent is replaced only if beaten by 5%.
- SKU override: a SKU keeps its own best model only if it beats the segment champion by 15% on at
  least three non-overlapping windows.
- Re-selection every four weeks.

Segments (as of the plan date, `demand/segmentation.py`): NEW_PRODUCT (< 26 weeks since launch),
INTERMITTENT (average demand interval ≥ 1.32 weeks), SEASONAL (Christmas-type), TRENDING
(|relative change over 26 seasonally adjusted weeks| ≥ 0.35), VOLATILE (CV² of non-zero weeks ≥
0.49), REGULAR.

**Challenger (as of the plan date, not used to plan).** statsmodels ETS(A, Ad, N) with smoothing,
trend and damping fitted by maximum likelihood, refitted at every fourth origin of the last 52
weeks and scored on exactly the champion's windows and scaled error (`forecasting/challengers.py`).
Over 1,732 windows on 184 SKUs its error was 0.394 against the champions' 0.306, and it was worse
in every segment: fitting parameters per SKU on one or two years of lumpy weekly demand overfits.
A global gradient-boosting challenger is on the roadmap, not in this version.

## Uncertainty

For the champion of each SKU, every scored origin gives a scaled error

    e = (actual − forecast) / (level × h)

where level is the larger of the SKU's mean weekly units over its last 26 usable weeks and over its
whole history, as of the origin. Errors are pooled by segment and horizon bucket (1, 2–4, 5–8,
9–13, 14–20, 21–26 weeks) and kept as 200 quantiles. Demand over h weeks is then
max(0, forecast + level × h × e).

Two earlier versions were rejected on the dev SKUs: ratio errors (actual / forecast) produced
80–400× tails for intermittent SKUs whose forecast was near zero before a large order, and a 26-week
level alone did the same for SKUs dormant for months.

**Calibration is measured, not assumed.** In the controlled replay QStats recorded its quantiles at
every weekly plan; the realised value uses the lead time that SKU's order would have had that week.
Over 6,560 SKU-weeks the P80 held 75.6% of outcomes, the P90 83.9% and the P95 89.1%. The intervals
are too narrow, most in the September–December season (P95: 84.7%) and for new products (77.1%);
REGULAR SKUs are close to nominal (P90 90.1%, P95 96.1%). Recalibration is the first roadmap item.

## Confidence

HIGH / MEDIUM / LOW from a documented score (`replenishment/recommendations.py`): history length,
segment (NEW_PRODUCT, INTERMITTENT, VOLATILE lower it), share of the last eight weeks that was
reconstructed, and the champion's backtest error. A purchase with LOW confidence is routed to
REVIEW_FORECAST instead of BUY.

## What the evidence says about accuracy

In the replay, QStats's one-week WAPE was 0.633, 0.627 and 0.625 in the three worlds against the
legacy process's 0.623, 0.628 and 0.626: no accuracy gain is claimed. Forecast bias fell from about
−20% to about −9% in all three, which is what stockout correction is for.
