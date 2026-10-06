# Moving from the demo data to a client's systems

The forecasting and planning code reads canonical tables only. Replacing the UCI file with a
client's data means writing one adapter, not changing the planner.

## The data contract (`domain/contract.py`)

| Table | Columns | Grain |
|---|---|---|
| `sales_lines` | order_id, order_ts, sku, description, quantity, unit_price, customer_id, country, line_type (SALE, RETURN, ADJUSTMENT, NON_MERCHANDISE) | order line |
| `demand_observations` | date, sku, location, units, price, promotion_flag, availability_status | SKU × location × day |
| `inventory_snapshots` | timestamp, sku, location, on_hand, available, reserved, inbound, transfer | SKU × location × day |
| `purchase_orders` | po_id, sku, supplier_id, location, quantity, order_date, expected_arrival, actual_arrival, status | PO line |
| `products` | sku, description, category, supplier_id, unit_cost, selling_price, case_pack, moq, cube_per_case, target_service_level, fba_enabled, preferred_source_dc | SKU |
| `suppliers` | supplier_id, supplier_name, country, quoted_lead_time_days, minimum_order_value, minimum_container_fill | supplier |
| `locations` | location_id, kind (DC, FBA, store, 3PL) | location |
| promotions | sku, start_date, end_date, kind, announced_date | event |

`validate_frame(df, table)` raises when a column is missing or has the wrong kind.

## Writing an adapter

Subclass `adapters.base.SourceAdapter` and return canonical frames:

```python
class AzureSQLAdapter(SourceAdapter):
    def __init__(self, url): self.engine = create_engine(url)
    def sales_lines(self):
        df = pd.read_sql(SALES_QUERY, self.engine)       # the client's column names live here only
        return validate_frame(rename_to_canonical(df), "sales_lines")
    def inventory_snapshots(self): ...
    def purchase_orders(self): ...
```

`adapters/csv_client.py` is a working example: a client exports the tables above as CSV with the
canonical column names and the pipeline runs on them (`tests/test_contract_and_api.py`).

## What changes, what does not

- **Unchanged:** cleaning rules, reconstruction, forecasting, lead times, safety stock, order
  quantities, FBA logic, recommendations, API, dashboard.
- **Replaced:** the simulation. A client's history already contains real inventory snapshots, POs
  and receipts; the planner runs on them directly (`PlannerView` is built from those tables).
- **To confirm with the client:** how their system marks cancellations and returns; whether
  inventory snapshots are end-of-day; what RESERVED means in their Amazon reports; supplier quotes;
  service targets by class; holding-cost rate.
- **Lost:** the controlled benchmark. Without a simulation there is no hidden true demand; the
  reconstruction is then checked with holdout tests (censor known-clean periods artificially and
  score the estimate), which use the same code.

## Database

Set `QSTATS_DATABASE_URL` to the client's PostgreSQL or Azure SQL database
(`postgresql+psycopg://…`, `mssql+pyodbc://…`) and install the driver. The schema in
`domain/tables.py` uses portable types only.
