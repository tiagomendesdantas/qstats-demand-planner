import numpy as np
import pandas as pd
import streamlit as st
import theme

import data

pal = theme.palette()
cfg = data.cfg()
scores = data.eval_table("benchmark_scores")
ep = data.eval_table("benchmark_episodes")
wk = data.eval_table("benchmark_weekly")
prods = data.table("products").set_index("sku")

theme.title(
    "Constrained demand recovery",
    "Sales are not demand when the shelf is empty. Here the real transaction history is treated as the demand a "
    "simulated business faced, and simulated stockouts cut sales short. Because the uncut series is kept aside, "
    "each reconstruction method can be scored against it: a controlled experiment, not a claim about the original "
    "retailer's lost sales.",
)

scores["channels"] = scores["channels"].fillna("all") if "channels" in scores else "all"
retro = scores[(scores["mode"] == "retrospective") & (scores["channels"] == "all")].set_index("method")
stocked = scores[(scores["mode"] == "retrospective") & (scores["channels"] != "all")].set_index("method")
none = retro.loc["no_adjustment"]
chosen = cfg["reconstruction"]["planner_method"]
c = retro.loc[chosen]
dark_units = none["lost_units"] - (stocked.at["no_adjustment", "lost_units"] if len(stocked) else none["lost_units"])
theme.strip(
    [
        ("Censored channel-days", theme.units(none["censored_days"]), f"{int(none['episodes']):,} stockout episodes"),
        ("True lost demand", theme.units(none["lost_units"]), "units the simulation hid"),
        ("Recovered by the planner", theme.pct(c["recovery_pct"], 0), f"{chosen.replace('_', ' ')}; net, see below"),
        ("Episode error", f"{c['episode_mae']:,.0f} u", f"vs {none['episode_mae']:,.0f} without adjustment"),
        ("Episode bias", f"{c['episode_bias']:+,.0f} u", "negative = still under-estimates"),
        ("On never-stocked channels", theme.pct(dark_units / none["lost_units"], 0), "of lost demand: invisible to every method"),
    ]
)

theme.section("Methods scored against the hidden baseline")
names = {
    "no_adjustment": "No adjustment (sales = demand)",
    "pre_post_velocity": "Pre/post velocity",
    "local_profile": "Local level × weekday × season",
    "model_expectation": "Smoothed level at episode start",
    "censored_gamma": "Censored likelihood (gamma, EM)",
}
scope = st.radio("Channels scored", ["all", "stocked at least once"], horizontal=True)
t = scores[scores["channels"] == scope].assign(
    Method=scores["method"].map(names), Mode=scores["mode"].map({"retrospective": "two-sided", "real_time": "real time"})
)
t["Used by planner"] = np.where(t["method"] == chosen, "✓", "")
t["recovery_pct"] = t["recovery_pct"] * 100
st.dataframe(
    t[
        ["Method", "Mode", "Used by planner", "episode_mae", "episode_bias", "weekly_mae", "recovery_pct", "daily_mae"]
    ].sort_values(["Mode", "episode_mae"]),
    hide_index=True,
    width="stretch",
    column_config={
        "episode_mae": st.column_config.NumberColumn("episode MAE (units)", format="%.0f"),
        "episode_bias": st.column_config.NumberColumn("episode bias", format="%+.0f"),
        "weekly_mae": st.column_config.NumberColumn("weekly MAE", format="%.1f"),
        "recovery_pct": st.column_config.NumberColumn("lost demand recovered", format="%.0f%%"),
        "daily_mae": st.column_config.NumberColumn("daily MAE", format="%.2f"),
    },
)
best = retro.drop("no_adjustment")["episode_mae"].idxmin()
theme.callout(
    f"The planner uses <b>{names[chosen]}</b>, chosen on 120 separate development SKUs by a rule fixed in advance "
    f"(lowest two-sided episode error). On these demo SKUs <b>{names[best]}</b> scored better "
    f"({retro.at[best, 'episode_mae']:,.0f} vs {c['episode_mae']:,.0f} units per episode, bias "
    f"{retro.at[best, 'episode_bias']:+,.0f} vs {c['episode_bias']:+,.0f})."
    if best != chosen
    else f"The planner uses <b>{names[chosen]}</b>, chosen on 120 separate development SKUs by a rule fixed in advance; "
    "it is also the best method on these demo SKUs."
)
biases = retro.drop("no_adjustment")["episode_bias"]
least = biases.abs().idxmin()
theme.note(
    (
        "Every method still under-estimates: stockouts tend to start on high-demand days, and the clean days around "
        "an episode are on average quieter than the days the shelf was empty. "
        if (biases < 0).all()
        else "The methods do not all err in the same direction. "
    )
    + "Only the censored-likelihood method uses the fact that on a sold-out day demand was at least what sold"
    + ("; it is the least biased here. " if least == "censored_gamma" else f"; the least biased here is {names[least]}. ")
    + "Recovered = (reconstructed − sold) on censored days ÷ true lost units: a net figure, in which over- and "
    "under-estimates on different days offset; episode error measures accuracy. Daily error barely moves: daily "
    "demand here is lumpy (most days nothing, some days a case or a wholesale order), so no estimate of the expected "
    "value matches a single day; the planner consumes weekly totals."
)

theme.section("One SKU: true demand, sales and reconstruction")
ep["stocked"] = ep["stocked"].astype(bool) if "stocked" in ep else True
top = ep.groupby(["sku", "channel"]).agg(baseline=("baseline", "sum"), stocked=("stocked", "first"))
top = top.sort_values("baseline", ascending=False).reset_index().head(40)
never = dict(zip(top["sku"] + "|" + top["channel"], ~top["stocked"], strict=True))
opts = [f"{r.sku}|{r.channel}" for r in top.itertuples()]
first_stocked = next((k for k, o in enumerate(opts) if not never[o]), 0)
pick = st.selectbox(
    "SKU and channel (most lost demand first)",
    opts,
    index=first_stocked,
    format_func=lambda o: (
        f"{o.split('|')[0]} · {prods.at[o.split('|')[0], 'description'].title()} · {o.split('|')[1].lower()} channel"
        + (" · never stocked" if never[o] else "")
    ),
)
sku, ch = pick.split("|")
if never[pick]:
    theme.callout(
        "This channel was never stocked after the warm-up, so its sales are zero every week and no method can see "
        "the demand: a channel that has never had stock looks exactly like a channel with no demand. Both planners "
        "share this blind spot; a channel-launch rule is on the roadmap."
    )
w = wk[(wk["sku"] == sku) & (wk["channel"] == ch)].copy()
w["week"] = pd.to_datetime(w["week"])
fig = theme.figure(340, y_title="units / week")
theme.shade_runs(fig, w["week"], w["censored_days"] > 0, pal["shade"], "stockout days")
fig.add_scatter(
    x=w["week"], y=w["baseline"], name="True demand (hidden from the planner)", line=dict(color=pal["ink"], width=1.5)
)
fig.add_scatter(x=w["week"], y=w["observed"], name="Observed sales", line=dict(color=pal["s2"], width=2))
fig.add_scatter(x=w["week"], y=w[chosen], name="Reconstructed (planner)", line=dict(color=pal["s1"], width=2))
if best != chosen:
    fig.add_scatter(
        x=w["week"],
        y=w[best],
        name=f"Reconstructed ({best.replace('_', ' ')})",
        line=dict(color=pal["s3"], width=1.5, dash="dot"),
    )
theme.show(fig)

theme.section("Episodes: true demand vs observed and reconstructed")
e = ep.copy()
fig = theme.figure(340, y_title="reconstructed or observed total", x_title="true demand during the episode")
mx = float(e["baseline"].quantile(0.99))
fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=pal["rule"], width=1), name="Perfect", hoverinfo="skip")
fig.add_scatter(
    x=e["baseline"],
    y=e["observed"],
    mode="markers",
    name="Observed sales",
    marker=dict(size=8, color=pal["s2"], opacity=0.55, line=dict(width=1, color=pal["surface"])),
)
fig.add_scatter(
    x=e["baseline"],
    y=e["reconstructed"],
    mode="markers",
    name="Reconstructed",
    marker=dict(size=8, color=pal["s1"], opacity=0.7, symbol="diamond", line=dict(width=1, color=pal["surface"])),
)
fig.update_xaxes(range=[0, mx], showgrid=True, gridcolor=pal["grid"])
fig.update_yaxes(range=[0, mx])
fig.update_layout(hovermode="closest")
theme.show(fig)
theme.note(
    f"{len(e):,} episodes (consecutive censored days of one SKU in one channel). Points on the diagonal are perfect; "
    "observed sales sit at or near zero by construction."
)
