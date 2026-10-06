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
the last observed receipt sits at the largest observed age, which is conservative.

## Demand over the protection interval

An order placed now must cover demand until the order placed at the next review arrives: lead time
L plus review period R (7 days). Both are uncertain, and they combine as an exact mixture, without
Monte Carlo:

    P(D ≤ d) = Σ_k p(L = l_k) · P( F(l_k + R) + level · (l_k + R)/7 · e ≤ d )

F(x) is the cumulative point forecast over the next x days (weekly forecasts spread over trading
days by the daily seasonal factor), e the pooled scaled errors at that horizon, and level the SKU's
error scale (see the forecasting methodology). The lead-time distribution is compressed to at most
16 support points; quantiles are read from the weighted mixture by inverted CDF, which is exact for
a mixture with discrete atoms.

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
demand, then from the other DC.

## Forward projection

    projected stock(d) = stock now + receipts expected by d − expected demand through d

Open POs land on their expected date (a DELAYED PO is assumed one week out); a recommended order
lands after the supplier's median lead time. The projected stockout date is the first day the
projection reaches zero. Expected lost units before relief come from a lost-sales recursion over the
same pipeline (`inventory/projection.expected_lost`), not from demand until a new order lands.

## Recommendations

| Action | Trigger |
|---|---|
| CRITICAL_STOCKOUT | out of stock now, or projected out within 14 days and before an order placed today could arrive |
| BUY | position below the order-up-to level |
| EXPEDITE | projected out more than 7 days before an open PO's expected arrival |
| TRANSFER | one DC under 2 weeks of cover while the other has over 8 |
| SEND_TO_FBA | Amazon position below its target and a DC can spare the units |
| EXCESS | more than 26 weeks of cover and over $500 beyond it |
| LOW_MARGIN | a BUY on a SKU whose contribution margin is under 15%: review, not buy |
| REVIEW_FORECAST | a BUY with LOW forecast confidence |
| STOCKOUT_CENSORED | over 20% of the last eight weeks' demand was reconstructed |

Each carries WHAT (action, quantity, location), WHY (one sentence from the numbers), EVIDENCE (the
inputs), EXPECTED EFFECT (cycle service before → after, or units short avoided) and CONFIDENCE.

## Economic impact

- Contribution protected by a purchase = (E[(D − position)⁺] − E[(D − position − order)⁺]) ×
  contribution per unit, over lead time + review.
- Critical stockouts and expedites: expected lost units before relief × contribution per unit.
- Transfers: cross-DC shipping avoided. Excess: yearly carrying cost at 24%.
- Portfolio figures (inventory value, contribution at risk, excess value, service now and after the
  plan) are in `economics/impact.py`. None is a hard-coded improvement.

## Containers

Purchase lines are grouped by supplier into 40ft high-cube containers (68 m³). A container is
topped up only when it is below the supplier's minimum fill (55%): to 65%, with whole cases of
that supplier's other SKUs, lowest cover first, never past 16 weeks of cover and never for a
low-margin or discontinued SKU. Filling a box for its own sake turns working capital into stock.
The greedy solver sits behind `ContainerSolver`, so an OR-Tools or PuLP model can replace it.
