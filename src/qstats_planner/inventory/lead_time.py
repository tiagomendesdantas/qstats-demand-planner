"""Supplier lead-time distributions.

Received purchase orders give complete lead times. Orders still open are right-censored: they
have taken at least their current age. Dropping them would bias the distribution short (late
orders are exactly the ones still open), so the distribution is a Kaplan-Meier estimate.

With few receipts the estimate is shrunk toward the supplier's quoted lead time: `k` pseudo-
observations spread around the quote (log-sd `prior_log_sd`) are added before estimating. Their
weight fades as real receipts accumulate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class LeadTimeDistribution:
    supplier_id: str
    days: np.ndarray       # support (integer days)
    prob: np.ndarray       # probabilities, sum 1
    n_received: int
    n_open: int
    quoted: float

    def quantile(self, q: float) -> float:
        c = np.cumsum(self.prob)
        return float(self.days[min(np.searchsorted(c, q - 1e-9), len(self.days) - 1)])

    @property
    def mean(self) -> float:
        return float((self.days * self.prob).sum())

    def scaled(self, multiplier: float = 1.0, spread: float = 1.0) -> LeadTimeDistribution:
        """Scenario helper: shift the distribution and/or stretch it around its median."""
        med = self.quantile(0.5)
        d = np.maximum(7, np.round((med + (self.days - med) * spread) * multiplier))
        return LeadTimeDistribution(self.supplier_id, d, self.prob, self.n_received, self.n_open, self.quoted * multiplier)


def kaplan_meier(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discrete distribution of event times. Mass left after the last event (censored tail) is
    placed at the largest observed time, which is conservative for a planner."""
    order = np.argsort(times, kind="stable")
    t, e = times[order], events[order]
    uniq = np.unique(t[e])
    surv = 1.0
    mass = []
    for u in uniq:
        at_risk = (t >= u).sum()
        d = ((t == u) & e).sum()
        p = surv * d / at_risk
        mass.append(p)
        surv -= p
    days = list(uniq)
    if surv > 1e-9:
        tail = max(t.max(), uniq.max() if len(uniq) else t.max())
        if days and tail == days[-1]:
            mass[-1] += surv
        else:
            days.append(tail)
            mass.append(surv)
    mass = np.array(mass)
    return np.array(days, float), mass / mass.sum()


def estimate(obs, supplier_id: str, quoted: float, k: int, prior_log_sd: float = 0.12) -> LeadTimeDistribution:
    o = obs[obs["supplier_id"] == supplier_id]
    times = o["lead_time_days"].to_numpy(float)
    events = o["event"].to_numpy(bool)
    # open orders younger than the quote carry almost no information about the tail
    keep = events | (times >= 0.5 * quoted)
    times, events = times[keep], events[keep]
    z = norm.ppf((np.arange(k) + 0.5) / k)
    pseudo = np.round(quoted * np.exp(prior_log_sd * z))
    t_all = np.concatenate([times, pseudo])
    e_all = np.concatenate([events, np.ones(k, bool)])
    days, prob = kaplan_meier(np.round(t_all), e_all)
    return LeadTimeDistribution(supplier_id, days, prob, int(events.sum()), int((~events).sum()), quoted)


def estimate_all(obs, suppliers, k: int) -> dict[str, LeadTimeDistribution]:
    return {r.supplier_id: estimate(obs, r.supplier_id, float(r.quoted_lead_time_days), k)
            for r in suppliers.itertuples()}


def point(quoted: float) -> LeadTimeDistribution:
    """The Legacy view: the quote, with certainty."""
    return LeadTimeDistribution("", np.array([quoted]), np.array([1.0]), 0, 0, quoted)
