"""Visual language: the "planner's workbook".

White canvas, hairline rules instead of cards, dense tables, one green-ink accent, IBM Plex.
Series colours were checked with the dataviz palette validator (all-pairs, light on #ffffff and
dark on #151614: lightness band, chroma, colour-blind separation, normal-vision floor, contrast).

    series 1  QStats / reconstructed / forecast   #1d7a55 light   #2f9c70 dark
    series 2  Legacy / observed sales             #2a78d6         #4a8fe0
    series 3  third series (Legacy + prior, ...)  #c98500         #c98500
    reference lines (true demand, safety stock) are drawn in ink, not a series colour.

Status colours (critical, serious, warning, good) are reserved for state and always come with a
glyph and a label.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

LIGHT = {
    "surface": "#ffffff",
    "panel": "#f4f5f2",
    "ink": "#1b1d1a",
    "ink2": "#4d514a",
    "muted": "#7a7f76",
    "rule": "#dcdfd8",
    "grid": "#eceee9",
    "accent": "#1d6b4a",
    "s1": "#1d7a55",
    "s2": "#2a78d6",
    "s3": "#c98500",
    "band": "rgba(29,122,85,0.13)",
    "band2": "rgba(29,122,85,0.07)",
    "shade": "rgba(27,29,26,0.06)",
    "event": "rgba(201,133,0,0.12)",
}
DARK = {
    "surface": "#151614",
    "panel": "#1e201d",
    "ink": "#e8eae5",
    "ink2": "#b9bdb4",
    "muted": "#8b9086",
    "rule": "#30332e",
    "grid": "#262925",
    "accent": "#4fb487",
    "s1": "#2f9c70",
    "s2": "#4a8fe0",
    "s3": "#c98500",
    "band": "rgba(47,156,112,0.22)",
    "band2": "rgba(47,156,112,0.11)",
    "shade": "rgba(232,234,229,0.07)",
    "event": "rgba(201,133,0,0.18)",
}
STATUS = {"CRITICAL": "#d03b3b", "HIGH": "#ec835a", "MEDIUM": "#c98500", "LOW": "#7a7f76", "INFO": "#7a7f76", "good": "#0ca30c"}
GLYPH = {"CRITICAL": "●", "HIGH": "▲", "MEDIUM": "■", "LOW": "○", "INFO": "·"}


def palette() -> dict:
    try:
        return DARK if st.context.theme.type == "dark" else LIGHT
    except Exception:
        return LIGHT


# ------------------------------------------------------------------ formatting


def money(v: float, digits: int = 0) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    a = abs(v)
    sign = "−" if v < 0 else ""
    if a >= 1e6:
        return f"{sign}${a / 1e6:,.2f}M"
    if a >= 1e4:
        return f"{sign}${a / 1e3:,.1f}k"
    return f"{sign}${a:,.{digits}f}"


def units(v: float) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:,.0f}"


def pct(v: float, digits: int = 1) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v * 100:.{digits}f}%"


def signed_pct(v: float, digits: int = 1) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{'+' if v >= 0 else '−'}{abs(v) * 100:.{digits}f}%"


def severity_label(s: str) -> str:
    return f"{GLYPH.get(s, '')} {s}"


# ------------------------------------------------------------------ page furniture

CSS = """
<style>
div.stMainBlockContainer.block-container {padding-top: 4.25rem !important;}
.qs-mast {border-top: 3px solid var(--qs-accent); padding: 10px 0 8px 0; margin: -8px 0 4px 0;
  display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid var(--qs-rule);}
.qs-brand {font-weight: 600; font-size: 15px; letter-spacing: .01em;}
.qs-brand span {color: var(--qs-muted); font-weight: 400;}
.qs-meta {font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--qs-muted);}
.qs-disclose {font-size: 12px; color: var(--qs-muted); margin: 2px 0 14px 0;}
.qs-title {font-size: 26px; font-weight: 600; margin: 6px 0 2px 0; letter-spacing: -.01em;}
.qs-lede {font-size: 15px; color: var(--qs-ink2); margin: 0 0 14px 0; max-width: 980px; line-height: 1.5;}
.qs-strip {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); border-top: 1px solid var(--qs-rule);
  border-bottom: 1px solid var(--qs-rule); margin: 6px 0 18px 0;}
.qs-cell {padding: 10px 14px 10px 0; border-right: 1px solid var(--qs-rule); margin-right: 14px;}
.qs-cell:last-child {border-right: none;}
.qs-lab {font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--qs-muted);}
.qs-val {font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 500; margin-top: 2px; white-space: nowrap;}
.qs-sub {font-size: 12px; color: var(--qs-ink2); margin-top: 1px;}
.qs-h {font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: .07em; color: var(--qs-ink2);
  border-bottom: 1px solid var(--qs-rule); padding-bottom: 4px; margin: 18px 0 8px 0;}
.qs-note {font-size: 12px; color: var(--qs-muted); line-height: 1.45; margin-top: 4px;}
.qs-chip {font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 500; white-space: nowrap;}
.qs-explain {border-left: 3px solid var(--qs-accent); padding: 4px 0 4px 14px; margin: 6px 0 10px 0;}
.qs-explain .what {font-size: 20px; font-weight: 600;}
.qs-explain .why {font-size: 14px; color: var(--qs-ink2); margin-top: 4px; line-height: 1.5;}
.qs-kv {display: grid; grid-template-columns: max-content 1fr; gap: 3px 18px; font-size: 13px; margin-top: 6px;}
.qs-kv .k {color: var(--qs-muted);} .qs-kv .v {font-family: 'IBM Plex Mono', monospace;}
.qs-callout {background: var(--qs-panel); padding: 10px 14px; font-size: 13.5px; line-height: 1.5; margin: 6px 0 12px 0;}
</style>
"""


def setup_page() -> dict:
    pal = palette()
    vars_ = ";".join(f"--qs-{k}:{v}" for k, v in pal.items())
    st.html(f"<style>:root{{{vars_}}}</style>" + CSS)
    return pal


def masthead(plan_date: str) -> None:
    st.html(
        f"<div class='qs-mast'><div class='qs-brand'>QStats <span>Demand &amp; Inventory Planner</span></div>"
        f"<div class='qs-meta'>plan date {plan_date} · weekly cycle · USD</div></div>"
        "<div class='qs-disclose'>Demand patterns: real transactions (UCI Online Retail II). Company, warehouses, "
        "Amazon FBA, suppliers, inventory and costs: simulated. See <b>About the data</b>.</div>"
    )


def title(text: str, lede: str | None = None) -> None:
    st.html(f"<div class='qs-title'>{text}</div>" + (f"<div class='qs-lede'>{lede}</div>" if lede else ""))


def section(text: str) -> None:
    st.html(f"<div class='qs-h'>{text}</div>")


def note(text: str) -> None:
    st.html(f"<div class='qs-note'>{text}</div>")


def callout(text: str) -> None:
    st.html(f"<div class='qs-callout'>{text}</div>")


def strip(cells: list[tuple[str, str, str]]) -> None:
    """A row of key numbers separated by hairlines: (label, value, sub-line)."""
    html = "".join(
        f"<div class='qs-cell'><div class='qs-lab'>{a}</div><div class='qs-val'>{b}</div><div class='qs-sub'>{c}</div></div>"
        for a, b, c in cells
    )
    st.html(f"<div class='qs-strip'>{html}</div>")


def kv(pairs: list[tuple[str, str]]) -> str:
    return "<div class='qs-kv'>" + "".join(f"<div class='k'>{k}</div><div class='v'>{v}</div>" for k, v in pairs) + "</div>"


def chip(severity: str) -> str:
    return f"<span class='qs-chip' style='color:{STATUS.get(severity, '#7a7f76')}'>{severity_label(severity)}</span>"


# ------------------------------------------------------------------ charts


def figure(height: int = 320, y_title: str | None = None, x_title: str | None = None) -> go.Figure:
    pal = palette()
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=12, color=pal["ink2"]),
        legend=dict(
            orientation="h",
            x=0,
            y=1.08,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12, color=pal["ink2"]),
        ),
        hoverlabel=dict(font=dict(family="IBM Plex Mono, monospace", size=12), bgcolor=pal["surface"], bordercolor=pal["rule"]),
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=pal["rule"],
        tickcolor=pal["rule"],
        ticks="outside",
        ticklen=3,
        automargin=True,
        title=dict(text=x_title, font=dict(size=12)) if x_title else None,
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor=pal["grid"],
        gridwidth=1,
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        automargin=True,
        title=dict(text=y_title, font=dict(size=12)) if y_title else None,
        tickformat=",~s",
    )
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False, "responsive": True})


def shade_runs(fig: go.Figure, x: np.ndarray, flags: np.ndarray, color: str, label: str, width_days: int = 7) -> None:
    """Shade contiguous runs of flagged weeks; the label is shown once."""
    import pandas as pd

    x = pd.to_datetime(pd.Series(x)).to_numpy()
    flags = np.asarray(flags, bool)
    shown = False
    i = 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j + 1 < len(flags) and flags[j + 1]:
                j += 1
            fig.add_vrect(
                x0=x[i],
                x1=x[j] + np.timedelta64(width_days, "D"),
                fillcolor=color,
                line_width=0,
                layer="below",
                annotation_text=label if not shown else None,
                annotation_position="top left",
                annotation_font=dict(size=11, color=palette()["muted"]),
            )
            shown = True
            i = j + 1
        else:
            i += 1
