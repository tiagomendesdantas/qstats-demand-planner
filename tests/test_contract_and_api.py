"""Canonical contract, the CSV client adapter, and the HTTP API."""

import shutil
import sqlite3

import pandas as pd
import pytest

from qstats_planner.adapters.csv_client import CSVClientAdapter
from qstats_planner.demand.cleaning import clean_sales_lines, demand_lines
from qstats_planner.domain.contract import ContractError, validate_frame
from qstats_planner.utils.config import ROOT
from tests.conftest import database_available


def test_contract_rejects_missing_or_mistyped_columns():
    good = pd.DataFrame({"location_id": pd.Series(["EAST_DC"], dtype="string"), "kind": pd.Series(["DC"], dtype="string")})
    assert validate_frame(good, "locations") is good
    with pytest.raises(ContractError):
        validate_frame(good.drop(columns="kind"), "locations")
    with pytest.raises(ContractError):
        validate_frame(
            pd.DataFrame(
                {
                    "date": ["x"],
                    "sku": ["a"],
                    "location": ["b"],
                    "units": ["many"],
                    "availability_status": ["NORMAL"],
                    "promotion_flag": [False],
                }
            ),
            "demand_observations",
        )


def test_csv_client_adapter_feeds_the_same_cleaning(tmp_path, cfg):
    pd.DataFrame(
        {
            "order_id": ["1", "2", "C3"],
            "order_ts": ["2025-01-02 10:00", "2025-01-03 11:00", "2025-01-04 09:00"],
            "sku": ["A-100", "A-100", "A-100"],
            "description": ["Mug", "Mug", "Mug"],
            "quantity": [5, 4, -4],
            "unit_price": [3.0, 3.0, 3.0],
            "customer_id": ["c1", "c2", "c2"],
            "country": ["US", "US", "US"],
            "line_type": ["SALE", "SALE", "RETURN"],
        }
    ).to_csv(tmp_path / "sales_lines.csv", index=False)
    lines = CSVClientAdapter(tmp_path).sales_lines()
    clean, rep = clean_sales_lines(lines, cfg)
    assert rep["cancelled_sale_rows"] == 1 and demand_lines(clean)["units"].sum() == 5
    assert CSVClientAdapter(tmp_path).products() is None


@pytest.fixture
def client(tmp_path, monkeypatch):
    if not database_available():
        pytest.skip("run `make demo` first: no database")
    db = tmp_path / "planner.sqlite"
    shutil.copy(ROOT / "data" / "planner.sqlite", db)
    with sqlite3.connect(db) as conn:  # decisions recorded in the app must not leak into the test
        conn.execute("DELETE FROM recommendation_overrides")
        conn.execute("UPDATE plan_recommendations SET status = 'OPEN'")
    monkeypatch.setenv("QSTATS_DATABASE_URL", f"sqlite:///{db}")
    from fastapi.testclient import TestClient

    from qstats_planner.api import main

    main._repo.cache_clear()
    yield TestClient(main.app)
    main._repo.cache_clear()


def test_api_read_endpoints(client):
    assert client.get("/health").json()["status"] == "ok"
    skus = client.get("/skus").json()
    assert len(skus) == 200
    sku = skus[0]["sku"]
    for path in (
        f"/skus/{sku}",
        f"/forecasts/{sku}",
        f"/inventory/{sku}",
        f"/recommendations/{sku}",
        "/recommendations?severity=CRITICAL",
        "/metrics",
        "/suppliers",
        "/containers",
    ):
        assert client.get(path).status_code == 200, path
    assert client.get("/skus/NOPE").status_code == 404
    assert client.get("/openapi.json").json()["info"]["title"].startswith("QStats")


def test_api_decisions_are_audited(client):
    rec = client.get("/recommendations?action=BUY&limit=1").json()[0]
    rid, qty = rec["recommendation_id"], rec["recommended_quantity"]
    r = client.post(f"/recommendations/{rid}/override", json={"quantity": qty + 24, "comment": "promo next month"})
    assert r.status_code == 200
    body = r.json()
    assert body["system_quantity"] == qty and body["override_quantity"] == qty + 24 and body["planner_action"] == "OVERRIDE"
    assert client.post(f"/recommendations/{rid}/override", json={"comment": "no quantity"}).status_code == 422
    assert client.post("/recommendations/REC-NOPE/accept").status_code == 404
    row = next(r for r in client.get(f"/recommendations/{rec['sku']}").json() if r["recommendation_id"] == rid)
    assert row["status"] == "OVERRIDDEN"
    assert len(client.get("/decisions").json()) == 1
