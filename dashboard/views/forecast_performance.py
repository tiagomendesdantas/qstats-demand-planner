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
fc = data.cfg()["forecasting"]
own = sp["selection_reason"].str.startswith("SKU's own best")
hsel = (sp["lt_p50"] + data.cfg()["simulation"]["review_period_days"]) / 7

theme.title(
    "Forecast performance",
    "Champion/challenger by rolling origin: every candidate forecasts from every past week "
    "using only data up to that week, scored on total demand over lead time + review, the quantity a purchase "
    "decision rests on. Each segment gets a champion; a SKU keeps its own pick only when it beats the segment's by "
    f"{fc['sku_override_margin']:.0%} on {fc['sku_override_min_blocks']} or more non-overlapping windows "
    f"({int(own.sum())} of {len(sp)} SKUs this week). Never a random split.",
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
    champion = segs.at[seg, "champion"] if seg in segs.index else "—"
    sc = segs.loc[seg].drop("champion") if seg in segs.index else pd.Series(dtype=float)
    sc = pd.to_numeric(sc, errors="coerce")
    best = sc.drop(labels=[champion], errors="ignore").dropna()
    rows.append(
        {
            "Segment": seg,
            "SKUs": len(g),
            "Champion": champion,
            "Champion error": sc.get(champion, np.nan),
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
    f"Scaled error = |forecast − actual| of total demand over the SKU's lead time + review ({hsel.min():.0f}–"
    f"{hsel.max():.0f} weeks this week, scored on the nearest of 6, 8, 11 or 14) ÷ (the SKU's mean weekly demand × "
    "weeks). "
    f"A simpler model within {fc['parsimony_margin']:.0%} of the best wins (parsimony); an incumbent is kept unless "
    f"beaten by {fc['switch_margin']:.0%}; a SKU gets its own model only if it beats the segment champion by "
    f"{fc['sku_override_margin']:.0%} on {fc['sku_override_min_blocks']} non-overlapping windows. NEW_PRODUCT SKUs "
    "have too few windows and use the pooled champion."
)

theme.section("Challenger: statsmodels ETS fitted per SKU")
ets = data.table("ets_challenger")
ets = ets[ets["windows"] > 0]
if len(ets):
    g = (
        ets.groupby("segment")
        .apply(
            lambda d: pd.Series(
                {
                    "SKUs": len(d),
                    "windows": int(d["windows"].sum()),
                    "champion error": (d["champion_error"] * d["windows"]).sum() / d["windows"].sum(),
                    "ETS error": (d["ets_error"] * d["windows"]).sum() / d["windows"].sum(),
                    "SKUs where ETS wins": (d["ets_error"] < d["champion_error"]).mean() * 100,
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    st.dataframe(
        g,
        hide_index=True,
        width="stretch",
        column_config={
            "champion error": st.column_config.NumberColumn(format="%.3f"),
            "ETS error": st.column_config.NumberColumn(format="%.3f"),
            "SKUs where ETS wins": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )
    tot_c = (ets["champion_error"] * ets["windows"]).sum() / ets["windows"].sum()
    tot_e = (ets["ets_error"] * ets["windows"]).sum() / ets["windows"].sum()
    theme.note(
        "ETS(A, Ad, N) with smoothing, trend and damping fitted by maximum likelihood at every fourth origin of the "
        "last 26 weeks, against the champion each SKU had 26 weeks before the plan date, so both are out of sample. "
        f"Same windows and scaled error: ETS {tot_e:.3f} vs champion {tot_c:.3f} over {int(ets['windows'].sum()):,} "
        f"windows on {len(ets)} SKUs. ETS did {'worse' if tot_e > tot_c else 'better'} overall, and "
        f"{'worse' if tot_e > tot_c else 'better'} in {int(((g['ETS error'] > g['champion error']) == (tot_e > tot_c)).sum())} "
        f"of {len(g)} segments. Reported, not used to plan."
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
    "A challenger can beat the champion on one SKU: a SKU's own windows decide only when there are at least "
    f"{fc['sku_override_min_blocks']} non-overlapping ones and the margin is {fc['sku_override_margin']:.0%}, "
    "because a handful of windows is too few to choose among 25 models without fitting noise."
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
peak95 = (cal.loc[cal["peak"], "realised"] <= cal.loc[cal["peak"], "q95"]).mean()
new = cal[cal["segment"] == "NEW_PRODUCT"] if "segment" in cal else cal.iloc[:0]
new95 = (new["realised"] <= new["q95"]).mean() if len(new) else float("nan")
theme.note(
    "From the controlled replay (World Q, all 200 SKUs, overlapping weekly windows, no interval computed): at every "
    "weekly plan QStats recorded its quantiles of demand over lead time + review; the outcome uses the lead time an "
    f"order placed that week would have had. Overall the P90 held {c90:.0%} of outcomes and the P95 {c95:.0%}"
    + (
        f": the intervals run narrow (P95: {peak95:.0%} in the September–December season, {new95:.0%} for new "
        "products). Two likely causes, both by construction: the errors come from the windows used to select the "
        "champion, and windows more than 10% reconstructed (busy stockout periods) are not scored. Recalibrating them "
        "is the first roadmap item."
        if c95 < 0.93
        else "."
    )
)
