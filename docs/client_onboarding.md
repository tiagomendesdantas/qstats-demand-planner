# Moving from the demo data to a client's systems

The forecasting and planning code reads canonical tables and a `PlannerView`; it does not import the
simulation (a test enforces this). Moving to a client's data means writing adapters and one
constructor, not changing the planner. This page separates what exists from what does not yet.

## The data contract (`domain/contract.py`)

| Table | Columns | Grain |
|---|---|---|
| `sales_lines` | order_id, order_ts, sku, description, quantity, unit_price, customer_id, country, line_type (SALE, RETURN, ADJUSTMENT, NON_MERCHANDISE) | order line |
| `demand_observations` | date, sku, location, units, availability_status, promotion_flag | SKU × location × day |
| `inventory_snapshots` | date, sku, location, on_hand, available, reserved, inbound, transfer | SKU × location × day |
| `purchase_orders` | po_id, sku, supplier_id, location, quantity, order_date, expected_arrival, status | PO line |
| `products` | sku, description, category, supplier_id, unit_cost, selling_price, case_pack, moq, cube_per_case, target_service_level, fba_enabled, preferred_source_dc | SKU |
| `suppliers` | supplier_id, supplier_name, country, quoted_lead_time_days, minimum_order_value | supplier |
| `locations` | location_id, kind (DC, FBA, store, 3PL) | location |
| `promotions` | sku, start_date, end_date, kind, announced_date | event |

`validate_frame(df, table)` raises when a column is missing or has the wrong kind.

## What exists

- **Adapters.** `adapters/uci.py` (the demo source) and `adapters/csv_client.py`: a client exports
  the tables above as CSV with the canonical column names and the adapter reads and validates them.
  The test in `tests/test_contract_and_api.py` runs client sales lines through the same cleaning
  rules as the UCI data.
- **Planner.** Reconstruction, forecasting, lead times, safety stock, order quantities, FBA logic,
  recommendations, API and dashboard all run on canonical inputs.
- **Database.** SQLAlchemy Core with portable types (`domain/tables.py`); `QSTATS_DATABASE_URL` points
  it at PostgreSQL (`postgresql+psycopg://…`) or Azure SQL (`mssql+pyodbc://…`) once the driver is
  installed. Only SQLite has been run.

## What a client deployment still needs

1. **A SQL adapter** per source system:

   ```python
   class AzureSQLAdapter(SourceAdapter):
       def __init__(self, url): self.engine = create_engine(url)
       def sales_lines(self):
           df = pd.read_sql(SALES_QUERY, self.engine)   # the client's column names live here only
           return validate_frame(rename_to_canonical(df), "sales_lines")
       def inventory_snapshots(self): ...
       def purchase_orders(self): ...
   ```

2. **A `PlannerView` built from the client's tables.** Today only the simulation engine constructs
   one. The constructor would assemble sales, snapshots (with their gaps), purchase orders with
   status as of the plan date, receipts and announced promotions from the canonical tables.
3. **A planning-cycle script** that reads that view instead of the simulated world.
4. **A reconstruction check without simulated truth.** With real data there is no hidden true
   demand. The check would censor known-clean periods artificially and score the estimate with
   `evaluation.benchmark.score`; the masking step is not written.

## To confirm with the client

How their system marks cancellations and returns; whether inventory snapshots are end-of-day; what
RESERVED means in their Amazon reports; supplier quotes and how late orders are recorded; service
targets (the demo's class targets did worse than a uniform 95%, see `EVAL_PLAN.md`); holding-cost
rate; which products are listed on Amazon but were never stocked there (the planner cannot see
demand on a channel that has never had stock).
