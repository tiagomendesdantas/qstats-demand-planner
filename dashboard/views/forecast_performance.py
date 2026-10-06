import numpy as np
import pandas as pd
import streamlit as st
import theme

import data

pal = theme.palette()
k = data.kpis()
perf = data.table("forecast_performance")
perf["champion"] = perf["champion"].astype(bool)
sp = data.table("sku_plan")
prods = data.table("products")
segs = data.table("segment_scores").set_index("segment")

theme.title(
    "Forecast performance",
    "Champion/challenger by rolling origin: every candidate forecasts from every past week "
    "using only data up to that week. The champion is chosen per segment on error of total demand over the lead "
    "time, the quantity a purchase decision rests on. Never a random split.",
)

champ = perf[perf["champion"]]
theme.strip(
    [
        ("WAPE, 1 week ahead", theme.pct(k["forecast_wape_26w"], 0), "champions, last 26 weeks"),
        ("Bias", theme.signed_pct(k["forecast_bias_26w"]), "sum of errors / sum of demand"),
        ("Candidates", f"{perf['model'].nunique()}", "baselines, smoothing, Croston family, ± seasonal prior"),
        ("Segments", f"{sp['segment'].nunique()}", ", ".join(sorted(sp["segment"].str.lower().str.replace("_", " ").unique()))),
    ]
)
theme.callout(
    "Measured on reconstructed demand (sales where stock was available, estimated demand where it was not), "
    "skipping windows that are more than 10% reconstructed. MAPE is not used: many weeks have zero or tiny demand."
)

theme.section("Champion by segment (scaled error of lead-time demand; lower is better)")
rows = []
for seg, g in sp.groupby("segment"):
    c = g["champion_model"].value_counts()
    sc = segs.loc[seg] if seg in segs.index else pd.Series(dtype=float)
    best = sc.drop(labels=[c.index[0]], errors="ignore").dropna()
    rows.append(
        {
            "Segment": seg,
            "SKUs": len(g),
            "Champion": c.index[0],
            "Champion error": sc.get(c.index[0], np.nan),
            "Naive error": sc.get("naive", np.nan),
            "Best other": best.idxmin() if len(best) else "—",
            "Best other error": best.min() if len(best) else np.nan,
            "SKUs on own model": int((g["selection_reason"].str.startswith("SKU's own")).sum()),
        }
    )
st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    width="stretch",
    column_config={
        c: st.column_config.NumberColumn(format="%.3f") for c in ["Champion error", "Naive error", "Best other error"]
    },
)
theme.note(
    "Scaled error = |forecast − actual| of total demand over 6–14 weeks ÷ (the SKU's mean weekly demand × weeks). "
    "A simpler model within 2% of the best wins (parsimony); an incumbent is kept unless beaten by 5%; a SKU gets its "
    "own model only if it beats the segment champion by 15% on three non-overlapping windows. NEW_PRODUCT SKUs have "
    "too few windows and use the pooled champion."
)

theme.section("By SKU")
c2 = champ.merge(sp[["sku_idx", "segment", "champion_model"]], on="sku_idx").merge(
    prods[["sku_idx", "description"]], on="sku_idx"
)
alt = perf[~perf["champion"]].sort_values("cum_scaled_error").drop_duplicates("sku_idx").set_index("sku_idx")
c2["best challenger"] = c2["sku_idx"].map(alt["model"])
c2["challenger error"] = c2["sku_idx"].map(alt["cum_scaled_error"])
st.dataframe(
    c2[
        [
            "sku",
            "description",
            "segment",
            "champion_model",
            "cum_scaled_error",
            "best challenger",
            "challenger error",
            "wape",
            "bias",
            "mae",
            "n",
        ]
    ],
    hide_index=True,
    width="stretch",
    height=360,
    column_config={
        "description": st.column_config.TextColumn("product", width="medium"),
        "cum_scaled_error": st.column_config.NumberColumn("champion error", format="%.2f"),
        "challenger error": st.column_config.NumberColumn(format="%.2f"),
        "wape": st.column_config.NumberColumn("WAPE 1-wk", format="%.2f"),
        "bias": st.column_config.NumberColumn(format="%.2f"),
        "mae": st.column_config.NumberColumn("MAE", format="%.1f"),
        "n": st.column_config.NumberColumn("weeks", format="%d"),
    },
)
theme.note(
    "A challenger can beat the champion on one SKU: the champion is chosen for the segment, because one SKU's "
    "two or three independent windows are too few to choose among 25 models without fitting noise."
)

theme.section("Backtest: one-week-ahead forecasts against what happened")
skus = c2.sort_values("n", ascending=False)["sku"].tolist()
sku = st.selectbox("SKU", skus, format_func=lambda s: f"{s} · {prods.set_index('sku').at[s, 'description'].title()}")
h = data.table("history", sku)
h["week"] = pd.to_datetime(h["week"])
h = h[h["week"] >= h["week"].max() - pd.Timedelta(weeks=52)]
fig = theme.figure(300, y_title="units / week")
theme.shade_runs(fig, h["week"], h["stockout_days"] > 0, pal["shade"], "stockout")
fig.add_scatter(
    x=h["week"],
    y=h["reconstructed"],
    name="Demand (reconstructed)",
    line=dict(color=pal["ink"], width=1.5),
    hovertemplate="%{y:,.0f}",
)
fig.add_scatter(
    x=h["week"],
    y=h["backtest_forecast"],
    name="Champion, forecast one week earlier",
    line=dict(color=pal["s1"], width=2),
    hovertemplate="%{y:,.0f}",
)
theme.show(fig)

theme.section("Are the intervals honest? Realised coverage")
cal = data.eval_table("calibration")
cal = cal[cal["seed"] == data.cfg()["random_seed"]]
cal["peak"] = cal["peak"].astype(bool)
qs = [("q50", 0.5), ("q80", 0.8), ("q90", 0.9), ("q95", 0.95)]
fig = theme.figure(280, y_title="share of outcomes at or below the quantile")
for name, mask, color, sym in (
    ("Outside Sep–Dec", ~cal["peak"], pal["s2"], "circle"),
    ("Sep–Dec peak", cal["peak"], pal["s3"], "diamond"),
):
    sub = cal[mask]
    cov = [(sub["realised"] <= sub[q]).mean() for q, _ in qs]
    fig.add_scatter(
        x=[f"P{int(v * 100)}" for _, v in qs],
        y=cov,
        name=f"{name} (n={len(sub):,})",
        mode="markers+lines",
        marker=dict(size=9, color=color, symbol=sym, line=dict(width=2, color=pal["surface"])),
        line=dict(color=color, width=1),
        hovertemplate="%{y:.1%}",
    )
fig.add_scatter(
    x=[f"P{int(v * 100)}" for _, v in qs],
    y=[v for _, v in qs],
    name="Nominal",
    mode="markers",
    marker=dict(size=14, symbol="line-ew", line=dict(width=2, color=pal["ink"])),
    hovertemplate="%{y:.0%}",
)
fig.update_yaxes(tickformat=".0%", range=[0.4, 1.0])
fig.update_layout(hovermode="closest")
theme.show(fig)
c90, c95 = (cal["realised"] <= cal["q90"]).mean(), (cal["realised"] <= cal["q95"]).mean()
theme.note(
    "From the controlled replay (World Q): at every weekly plan QStats recorded its quantiles of demand over lead "
    "time + review; the outcome uses the lead time that SKU's order would actually have had that week. "
    f"Overall the P90 held {c90:.0%} of outcomes and the P95 {c95:.0%}"
    + (
        ": the intervals run narrow, most in the peak season. Reported, not hidden; recalibrating them is on the roadmap."
        if c95 < 0.93
        else "."
    )
)
