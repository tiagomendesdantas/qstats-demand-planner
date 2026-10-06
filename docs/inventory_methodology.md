# Inventory methodology

## Inventory position

    position = on hand at every location − (1 − w) × Amazon RESERVED
               + units in transit to Amazon + open purchase orders (OPEN, IN_TRANSIT, DELAYED)

Units picked for an Amazon transfer stay in the DC's on-hand (they are committed, not sellable)
until they ship. Inventory states are not equal: AVAILABLE and Amazon FC TRANSFER count in full,
RESERVED at w = 0.5 because part of it never returns to sellable (the simulation writes a share
off). The legacy rule counts every unit at w = 1.

## Lead time

Kaplan–Meier over each supplier's purchase orders: received orders give complete lead times, open
orders older than half the quote are right-censored at their age (late orders are exactly the ones
still open, so dropping them would bias the distribution short). With few receipts the estimate is
shrunk toward the quote with eight pseudo-observations spread around it (log-sd 0.12). Mass beyond
the last observed receipt (orders still open past every receipt) sits at the largest observed age:
a lower bound for those orders, so the far tail is, if anything, understated.

## Demand over the protection interval

An order placed now must cover demand until the order placed at the next review arrives: lead time
L plus review period R (7 days). Both are uncertain, and they combine as a mixture computed without
simulation:

    P(D ≤ d) = Σ_k p(L = l_k) · P( F(l_k + R) + level · (l_k + R)/7 · e ≤ d )

F(x) is the cumulative point forecast over the next x days (weekly forecasts spread over trading
days by the daily seasonal factor), e the pooled scaled errors at that horizon, and level the SKU's
error scale (see the forecasting methodology). Quantiles are read from the weighted mixture by
inverted CDF. Approximations: the lead-time distribution is compressed to at most 16 support points,
errors are stored as 200 quantiles per segment and horizon bucket (one representative horizon per
bucket), and lead time is assumed independent of forecast error.

    order-up-to level   S  = Q_α(D)                 α = the product's cycle service target
    safety stock        SS = Q_α(D) − E[D]
    reorder point       S, recomputed every week

With periodic review the reorder point and the order-up-to level are the same quantity: an order
is placed when the position is below S. Both move every week with the forecast, the error
distribution and the supplier's observed lead times; nothing is a fixed number of days.

A textbook normal approximation, SS = z·sqrt(L σ_d² + d² σ_L²), is kept in
`inventory/lead_time_demand.py` as a cross-check; it is not used.

## Order quantity

    raw requirement   = S − position              (when position < S)
    order             = max(raw, MOQ), rounded UP to whole case packs

Example (`tests/test_quantities_and_inventory.py`): raw 1,047, case pack 24, MOQ 600 → 1,056.
Rounding up keeps the service target; when MOQ or the case pack inflate an order by more than 25%
the recommendation says so. The order is split between the two DCs in whole cases by each DC's
share of recent demand (Amazon demand counted at the DC that supplies Amazon).

## Amazon FBA

Planned on its own: Amazon demand = network forecast × Amazon's share of the last 13 weeks of
reconstructed demand; the replenishment window = pick (2 days) + transit (empirical distribution of
past transfers, shrunk toward the 5–12 day range) + review. Target = the service-level quantile of
Amazon demand over that window; position = available + FC transfer + inbound + units being picked
+ 0.5 × reserved. The transfer comes from the preferred DC, keeping a week of that DC's own expected
demand (`dc_protection_days_for_fba`), then from the other DC.

When no Amazon demand is forecast (under one unit per hundred days), the target is the forecast-
error allowance alone, because errors are scaled by the SKU's longer-run demand. Such a send, one
that avoids less than one unit of expected shortfall, or one whose Amazon contribution per unit is
negative is still listed but marked LOW, and its reason says why.

## Forward projection

    projected stock(d) = stock now + receipts expected by d − expected demand through d

Open POs land on their expected date (a DELAYED PO is assumed one week out); this week's recommended
order lands after the supplier's median lead time. Orders from later weekly reviews are not in the
projection, so a line that still reaches zero months out is expected: next week's plan orders again. The projected stockout date is the first day the
projection reaches zero. Expected lost units before relief come from a lost-sales recursion over the
same pipeline (`inventory/projection.expected_lost`), not from demand until a new order lands.

## Recommendations

| Action | Trigger |
|---|---|
| CRITICAL_STOCKOUT | out of stock now, or projected out within 14 days and before an order placed today could arrive |
| BUY | position below the order-up-to level |
| EXPEDITE | projected out more than 7 days before an open PO's expected arrival |
| TRANSFER | one DC under 2 weeks of cover while the other has over 8, and the moved quantity (whole cases) stays within 26 weeks of the receiving DC's demand |
| SEND_TO_FBA | Amazon position below its target and a DC can spare the units |
| EXCESS | stock + open orders beyond 26 weeks of forecast demand, over $500 beyond it (all of it when no demand is forecast) |
| LOW_MARGIN | a BUY on a SKU whose contribution margin is under 15%: review, not buy |
| REVIEW_FORECAST | a BUY with LOW forecast confidence |
| STOCKOUT_CENSORED | over 20% of the last eight weeks' demand was reconstructed |

Each carries WHAT (action, quantity, location), WHY (one sentence from the numbers), EVIDENCE (the
inputs), EXPECTED EFFECT (cycle service before → after, or units short avoided) and CONFIDENCE.
Purchase lines are not weighed against carrying cost: an order that MOQ or case packs inflate is
flagged in its reason when the rounding adds more than 25%, but it is not sent for review. A
forecast under one unit per hundred days counts as no forecast demand: reasons say so instead of
printing a ratio such as weeks of cover.

## Economic impact

Every money figure attached to an action comes from one timing-aware projection: stock today,
open POs landing on their expected dates (a DELAYED one a week out), this week's orders after the
supplier's median lead time, demand at its daily forecast, unmet demand lost
(`inventory/projection.expected_lost`). Because demand is held at its forecast, the projection
leaves out variability: it understates both the losses ahead and what safety stock buys.

- Purchases: demand the order serves from its arrival until the next weekly order can land (one
  review period later) that would otherwise be lost, × contribution per unit. Demand before it
  lands is not credited to it.
- Critical stockouts: demand lost before an order placed today could land × contribution per unit,
  booked as a negative value (a cost of the current position, like an EXCESS line's carrying cost).
  With purchases credited only after they land, a SKU's CRITICAL loss and its purchase value never
  count the same units.
- Expedites: demand the pulled-forward PO would serve before its current date, × contribution.
- Transfers: cross-DC shipping avoided. Excess: yearly carrying cost at 24%. Amazon sends: the
  reduction in expected Amazon shortfall over the replenishment window × Amazon contribution.
- Portfolio figures (`economics/impact.py`): contribution at risk is the projection's lost demand
  over the next 13 weeks × contribution; projected fill is the share of the next 13 weeks' forecast
  demand served, now and with this week's orders. The excess KPI counts physical stock on hand
  beyond 26 weeks of forecast demand; the EXCESS line counts stock and open orders, because delaying
  or cancelling an order is one of its remedies. None is a hard-coded improvement.

**Cycle service is a different quantity.** P(demand over lead time + review ≤ inventory position)
is what orders are sized to (the order-up-to level), and the SKU pages show it before and after an
order. It is a backorder view: an open PO counts whenever it arrives, so a SKU that is out of stock
today can score high. It is not a forecast of fill and is not used for the money figures.

## Containers

BUY lines are grouped by supplier into 40ft high-cube containers (68 m³); lines awaiting review
(LOW_MARGIN, REVIEW_FORECAST) are not packed until a person approves them. A container is topped up
only when it is below the supplier's minimum fill (55%): to 65%, with whole cases of that
supplier's other SKUs, lowest cover first, as far as eligible SKUs allow: never past 16 weeks of
cover (on hand + on order + this order) and never for a low-margin, discontinued or low-confidence
SKU. A container that still falls short of the minimum is flagged. Filling a box for its own sake
turns working capital into stock. The Scenario simulator calls the same routine, so its base
run shows the same containers as this page.
The greedy solver sits behind `ContainerSolver`, so an OR-Tools or PuLP model can replace it.
