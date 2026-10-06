"""SKU segmentation from interpretable features, computed as of the plan date.

Order of precedence (the first rule that matches wins):
    NEW_PRODUCT    fewer than `new_product_weeks` weeks since launch
    INTERMITTENT   average demand interval >= 1.32 weeks (Syntetos-Boylan)
    SEASONAL       Christmas-type product (description keyword; the seasonal prior's group)
    TRENDING       |relative change over the last 26 weeks, seasonally adjusted| >= threshold
    VOLATILE       squared CV of non-zero weekly demand >= 0.49 (Syntetos-Boylan "erratic")
    REGULAR        everything else
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qstats_planner.demand.features import trend_strength

SEGMENTS = ("REGULAR", "INTERMITTENT", "SEASONAL", "TRENDING", "NEW_PRODUCT", "VOLATILE")


def segment_features(Y: np.ndarray, valid: np.ndarray, y_adj: np.ndarray, weeks_since_launch: np.ndarray) -> pd.DataFrame:
    """Y units (W, n); y_adj seasonally adjusted rate (W, n)."""
    n = Y.shape[1]
    rows = []
    for i in range(n):
        v = valid[-52:, i]
        x = Y[-52:, i][v]
        nz = x[x > 0]
        adi = len(x) / len(nz) if len(nz) else np.inf
        cv2 = float(np.var(nz) / np.mean(nz) ** 2) if len(nz) > 1 else 0.0
        r = y_adj[-26:, i][valid[-26:, i]]
        rows.append({
            "weeks_since_launch": int(weeks_since_launch[i]), "history_weeks": int(valid[:, i].sum()),
            "adi": adi, "cv2": cv2, "zero_week_ratio": float((x == 0).mean()) if len(x) else 1.0,
            "mean_weekly_units": float(x.mean()) if len(x) else 0.0,
            "trend_26w": trend_strength(np.nan_to_num(r)) if len(r) >= 8 else 0.0,
        })
    return pd.DataFrame(rows)


def assign_segments(feat: pd.DataFrame, seasonal_group: np.ndarray, cfg: dict) -> np.ndarray:
    s = cfg["segmentation"]
    return np.select(
        [
            feat["weeks_since_launch"] < s["new_product_weeks"],
            feat["adi"] >= s["adi_threshold"],
            seasonal_group.astype(bool),
            feat["trend_26w"].abs() >= s["trend_threshold"],
            feat["cv2"] >= s["cv2_threshold"],
        ],
        ["NEW_PRODUCT", "INTERMITTENT", "SEASONAL", "TRENDING", "VOLATILE"],
        default="REGULAR",
    )
