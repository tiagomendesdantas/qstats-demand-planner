"""HTTP API over the planner database.

    uvicorn qstats_planner.api.main:app --reload       # docs at /docs (Swagger) and /redoc

Read endpoints return the current plan; the three POST endpoints record a planner decision
(accept, override with a quantity, reject) next to the system's original recommendation.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from qstats_planner.services.repository import Repository
from qstats_planner.utils.config import database_url, load_config

app = FastAPI(
    title="QStats Demand & Inventory Planner API",
    version="0.1.0",
    description="Recommendations, forecasts, inventory and evidence from the planning cycle. Demo data: real demand "
    "patterns (UCI Online Retail II) in a simulated supply chain.",
)


@lru_cache(maxsize=1)
def _repo() -> Repository:
    return Repository(database_url(load_config()))


def repo() -> Repository:
    return _repo()


def _clean(v: Any) -> Any:
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) or math.isinf(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return v.isoformat()
    return v


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


class Decision(BaseModel):
    quantity: int | None = Field(None, ge=0, description="Required for an override")
    comment: str = ""
    planner: str = "planner"


class DecisionResult(BaseModel):
    recommendation_id: str
    sku: str
    action: str
    system_quantity: int
    planner_action: Literal["ACCEPT", "OVERRIDE", "REJECT"]
    override_quantity: int
    comment: str
    planner: str
    timestamp: datetime


@app.get("/health", tags=["meta"])
def health(r: Repository = Depends(repo)) -> dict:
    k = r.kpis()
    return {"status": "ok", "plan_date": k.get("plan_date"), "skus": k.get("skus_monitored")}


@app.get("/skus", tags=["products"])
def skus(r: Repository = Depends(repo)) -> list[dict]:
    p = r.products().merge(r.sku_plan().drop(columns=["sku"], errors="ignore"), on="sku_idx")
    cols = [
        "sku",
        "description",
        "category",
        "supplier_id",
        "segment",
        "champion_model",
        "confidence",
        "inventory_position",
        "on_order",
        "weekly_demand",
        "weeks_of_cover",
        "recommended_quantity",
        "target_service_level",
    ]
    return records(p[cols])


@app.get("/skus/{sku}", tags=["products"])
def sku_detail(sku: str, r: Repository = Depends(repo)) -> dict:
    prod = r.product(sku)
    if prod is None:
        raise HTTPException(404, f"unknown SKU {sku}")
    plan = r.sku_plan()
    row = plan[plan["sku_idx"] == prod["sku_idx"]]
    return {
        "product": {k: _clean(v) for k, v in prod.items()},
        "plan": records(row)[0] if len(row) else None,
        "recommendations": records(r.recommendations(sku=sku)),
    }


@app.get("/forecasts/{sku}", tags=["forecasts"])
def forecasts(sku: str, history_weeks: int = Query(52, ge=0, le=200), r: Repository = Depends(repo)) -> dict:
    if r.product(sku) is None:
        raise HTTPException(404, f"unknown SKU {sku}")
    h = r.history(sku)
    return {
        "sku": sku,
        "forecast": records(r.forecast(sku)),
        "history": records(h.tail(history_weeks)),
        "backtest": records(r.forecast_performance(sku)),
    }


@app.get("/inventory/{sku}", tags=["inventory"])
def inventory(sku: str, r: Repository = Depends(repo)) -> dict:
    if r.product(sku) is None:
        raise HTTPException(404, f"unknown SKU {sku}")
    inv = r.inventory(sku)
    latest = inv[inv["date"] == inv["date"].max()] if len(inv) else inv
    return {
        "sku": sku,
        "latest": records(latest),
        "projection": records(r.projection(sku)),
        "purchase_orders": records(r.purchase_orders(sku)),
    }


@app.get("/recommendations", tags=["recommendations"])
def recommendations(
    action: str | None = None,
    severity: str | None = None,
    supplier_id: str | None = None,
    category: str | None = None,
    location: str | None = None,
    segment: str | None = None,
    status: str | None = None,
    limit: int = Query(500, ge=1, le=5000),
    r: Repository = Depends(repo),
) -> list[dict]:
    df = r.recommendations(
        action=action,
        severity=severity,
        supplier_id=supplier_id,
        category=category,
        location=location,
        segment=segment,
        status=status,
    ).head(limit)
    return _with_evidence(df)


def _with_evidence(df: pd.DataFrame) -> list[dict]:
    out = records(df)
    for o in out:
        o["evidence"] = json.loads(o["evidence"]) if o.get("evidence") else {}
    return out


@app.get("/recommendations/{sku}", tags=["recommendations"])
def recommendations_for_sku(sku: str, r: Repository = Depends(repo)) -> list[dict]:
    if r.product(sku) is None:
        raise HTTPException(404, f"unknown SKU {sku}")
    return _with_evidence(r.recommendations(sku=sku))


def _decide(rec_id: str, action: str, body: Decision, r: Repository) -> DecisionResult:
    try:
        row = r.record_decision(rec_id, action, body.quantity, body.comment, body.planner)
    except KeyError:
        raise HTTPException(404, f"unknown recommendation {rec_id}") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    return DecisionResult(**row)


@app.post("/recommendations/{rec_id}/accept", response_model=DecisionResult, tags=["decisions"])
def accept(rec_id: str, body: Decision | None = None, r: Repository = Depends(repo)) -> DecisionResult:
    return _decide(rec_id, "ACCEPT", body or Decision(), r)


@app.post("/recommendations/{rec_id}/override", response_model=DecisionResult, tags=["decisions"])
def override(rec_id: str, body: Decision, r: Repository = Depends(repo)) -> DecisionResult:
    return _decide(rec_id, "OVERRIDE", body, r)


@app.post("/recommendations/{rec_id}/reject", response_model=DecisionResult, tags=["decisions"])
def reject(rec_id: str, body: Decision | None = None, r: Repository = Depends(repo)) -> DecisionResult:
    return _decide(rec_id, "REJECT", body or Decision(), r)


@app.get("/decisions", tags=["decisions"])
def decisions(r: Repository = Depends(repo)) -> list[dict]:
    return records(r.decisions())


@app.get("/metrics", tags=["metrics"])
def metrics(r: Repository = Depends(repo)) -> dict:
    k = r.kpis()
    try:
        matched = r.eval_meta().get("matched", {})
    except Exception:
        matched = {}
    return {"plan": k, "legacy_vs_qstats": matched}


@app.get("/suppliers", tags=["suppliers"])
def suppliers(r: Repository = Depends(repo)) -> list[dict]:
    return records(r.suppliers())


@app.get("/containers", tags=["containers"])
def containers(r: Repository = Depends(repo)) -> dict:
    lines, summary = r.containers()
    return {"summary": records(summary), "lines": records(lines)}
