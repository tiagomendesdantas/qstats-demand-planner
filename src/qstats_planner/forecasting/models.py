"""Forecast models as vectorised recursions over (weeks x SKUs).

Every model works on the seasonally adjusted rate per trading day,

    y_t = x_t / (e_t * s_t)        x units, e trading days in the week, s seasonal prior (or 1)

and states its forecast as a level plus an optional damped trend, so one pass over the history
gives the forecast from every origin at once (that is what makes rolling-origin backtests cheap):

    rate_{t+h} = level_t + trend_t * (phi + phi^2 + ... + phi^h)
    units_{t+h} = rate_{t+h} * e_{t+h} * s_{t+h}

Weeks marked invalid (before launch, no trading days, promotion or liquidation weeks) do not
update a model's state. Seasonal naive is the exception: it copies the same week last year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str  # naive | ma | ses | holt | croston | sba | tsb | snaive
    params: tuple
    complexity: int  # used by the parsimony rule (lower = simpler)
    seasonal: bool  # uses the seasonal prior


def candidate_models(cfg: dict, with_prior: bool = True) -> list[ModelSpec]:
    f = cfg["forecasting"]
    base = [ModelSpec("naive", "naive", (), 0, False)]
    base += [ModelSpec(f"ma{k}", "ma", (k,), 1, False) for k in f["ma_windows"]]
    base += [ModelSpec(f"ses{a:.1f}", "ses", (a,), 2, False) for a in f["ses_alphas"]]
    a = f["croston_alpha"]
    base += [
        ModelSpec("croston", "croston", (a,), 3, False),
        ModelSpec("sba", "sba", (a,), 3, False),
        ModelSpec("tsb", "tsb", tuple(f["tsb_params"]), 3, False),
    ]
    base += [ModelSpec(f"holt_damped{k + 1}", "holt", tuple(p), 4, False) for k, p in enumerate(f["holt_params"])]
    out = list(base)
    if with_prior:
        out += [ModelSpec(m.name + "+prior", m.family, m.params, m.complexity + 1, True) for m in base]
        out.append(ModelSpec("seasonal_naive", "snaive", (52,), 5, True))
    return out


@dataclass
class States:
    level: np.ndarray  # (M, W, n) state after each week
    trend: np.ndarray  # (M, W, n)
    phi: np.ndarray  # (M,)


def run_states(models: list[ModelSpec], y_plain: np.ndarray, y_seas: np.ndarray, valid: np.ndarray) -> States:
    """Run every model over every week. y_* are (W, n) adjusted rates (NaN where invalid)."""
    W, n = y_plain.shape
    M = len(models)
    level = np.full((M, W, n), np.nan)
    trend = np.zeros((M, W, n))
    phi = np.ones(M)
    for m, spec in enumerate(models):
        y = y_seas if spec.seasonal else y_plain
        fam = spec.family
        if fam == "snaive":
            continue  # handled at forecast time
        if fam == "naive":
            lv = np.full(n, np.nan)
            for t in range(W):
                lv = np.where(valid[t], y[t], lv)
                level[m, t] = lv
        elif fam == "ma":
            k = spec.params[0]
            buf = np.full((k, n), np.nan)
            cnt = np.zeros(n, int)
            for t in range(W):
                v = valid[t]
                buf = np.where(v[None, :], np.vstack([buf[1:], y[t][None, :]]), buf)
                cnt = np.where(v, np.minimum(cnt + 1, k), cnt)
                level[m, t] = np.where(cnt > 0, np.nanmean(np.where(np.isnan(buf), np.nan, buf), axis=0), np.nan)
        elif fam == "ses":
            a = spec.params[0]
            lv = np.full(n, np.nan)
            for t in range(W):
                v = valid[t]
                first = v & np.isnan(lv)
                lv = np.where(first, y[t], np.where(v, a * y[t] + (1 - a) * lv, lv))
                level[m, t] = lv
        elif fam == "holt":
            a, b, ph = spec.params
            phi[m] = ph
            lv = np.full(n, np.nan)
            tr = np.zeros(n)
            seen = np.zeros(n, int)
            for t in range(W):
                v = valid[t]
                first = v & (seen == 0)
                prev = lv
                new_l = a * y[t] + (1 - a) * (prev + ph * tr)
                new_t = b * (new_l - prev) + (1 - b) * ph * tr
                lv = np.where(first, y[t], np.where(v, new_l, lv))
                tr = np.where(first, 0.0, np.where(v & (seen > 0), new_t, tr))
                seen = seen + v
                level[m, t] = lv
                trend[m, t] = tr
        elif fam in ("croston", "sba"):
            a = spec.params[0]
            z = np.full(n, np.nan)
            p = np.full(n, np.nan)
            q = np.zeros(n)  # periods since the last demand
            for t in range(W):
                v = valid[t]
                q = q + v
                pos = v & (y[t] > 0)
                first = pos & np.isnan(z)
                z = np.where(first, y[t], np.where(pos, a * y[t] + (1 - a) * z, z))
                p = np.where(first, q, np.where(pos, a * q + (1 - a) * p, p))
                q = np.where(pos, 0, q)
                f = z / p
                level[m, t] = f * (1 - a / 2) if fam == "sba" else f
        elif fam == "tsb":
            a, b = spec.params
            z = np.full(n, np.nan)
            pr = np.full(n, np.nan)
            for t in range(W):
                v = valid[t]
                pos = y[t] > 0
                first_v = v & np.isnan(pr)
                pr = np.where(first_v, pos.astype(float), np.where(v, b * pos + (1 - b) * pr, pr))
                first_z = v & pos & np.isnan(z)
                z = np.where(first_z, y[t], np.where(v & pos, a * y[t] + (1 - a) * z, z))
                level[m, t] = np.where(np.isnan(z), np.where(np.isnan(pr), np.nan, 0.0), pr * z)
        # no history yet: NaN stays NaN
    return States(level, trend, phi)


def phi_sums(phi: np.ndarray, H: int) -> np.ndarray:
    """(M, H) cumulative damped-trend multipliers phi + ... + phi^h."""
    h = np.arange(1, H + 1)
    return np.cumsum(phi[:, None] ** h[None, :], axis=1)


def weekly_paths(
    states: States,
    models: list[ModelSpec],
    origin: int,
    H: int,
    es_plain: np.ndarray,
    es_seas: np.ndarray,
    units_hist: np.ndarray,
    exposure_hist: np.ndarray,
) -> np.ndarray:
    """(M, H, n) unit forecasts for weeks origin+1 .. origin+H.

    es_* are (H, n) exposure x season for those weeks (season = 1 for the plain variant).
    """
    M = len(models)
    n = units_hist.shape[1]
    out = np.full((M, H, n), np.nan)
    ps = phi_sums(states.phi, H)
    for m, spec in enumerate(models):
        es = es_seas if spec.seasonal else es_plain
        if spec.family == "snaive":
            for h in range(H):
                w = origin + 1 + h - 52
                if 0 <= w < units_hist.shape[0]:
                    e_then = exposure_hist[w]
                    out[m, h] = np.where(e_then > 0, units_hist[w] * es_plain[h] / max(e_then, 1e-9), np.nan)
            continue
        lv = states.level[m, origin]
        tr = states.trend[m, origin]
        rate = lv[None, :] + tr[None, :] * ps[m][:, None]
        out[m] = np.maximum(rate, 0) * es
    return out
