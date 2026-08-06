"""
ENT 305 — Session 11 · Thermal-Time Reasoning
Forward accumulation.

Per the session script this is NOT projected until minute 55. Students do the
accumulation by hand first. When it appears, the point is the board line:
    THE APP COMPUTES. IT DOES NOT KNOW.

Run:  streamlit run ent305_s11_forward_accumulation.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ent305_common import (DEV_REFERENCES, GOLD, MUTED, PLOT_LAYOUT, RUST,
                           SLATE, TEAL_D, TEAL_M, forward_accumulate,
                           make_temperature_record)

st.set_page_config(page_title="ENT 305 · Forward accumulation", layout="wide")

st.title("Forward accumulation")
st.caption("Session 11 · Thermal-Time Reasoning — how much development does a "
           "temperature history buy?")

# ---------------------------------------------------------------- data
with st.sidebar:
    st.header("Temperature record")
    src = st.radio("Source", ["Built-in record", "Upload CSV"], index=0)
    if src == "Upload CSV":
        up = st.file_uploader("Two columns: Date/Time, Temp_C", type="csv")
        if up is None:
            st.info("Using the built-in record until a file is uploaded.")
            df = make_temperature_record()
        else:
            df = pd.read_csv(up)
            df["Date/Time"] = pd.to_datetime(df["Date/Time"], errors="coerce")
            bad = int(df["Date/Time"].isna().sum())
            if bad:
                st.warning(f"{bad} row(s) have an unreadable Date/Time and are "
                           f"shown below but excluded from the accumulation.")
            df = df.dropna(subset=["Date/Time"])
    else:
        cold = st.checkbox("Include a cold stretch on day 3", value=False,
                           help="A cold night makes the walk longer without "
                                "banking anything.")
        df = make_temperature_record(cold_night=(3, 7.0) if cold else None)

    st.markdown("---")
    st.header("Your assumptions")
    t_base = st.slider("Base temperature, Tbase (°C)", 4.0, 16.0, 10.0, 0.5,
                       help="Below this, development banks zero. This is an "
                            "assumption, not a measurement.")
    stage = st.selectbox("Target stage", list(DEV_REFERENCES.keys()), index=3)
    default_adh = DEV_REFERENCES[stage]
    target_adh = st.slider("Requirement to reach it (degree-hours)",
                           50, 3000, int(default_adh), 10,
                           help="Published figures come from another "
                                "laboratory, another diet, constant "
                                "conditions. Move it and watch.")
    unit = st.radio("Display units", ["Degree-hours (ADH)", "Degree-days (ADD)"],
                    index=0, horizontal=False)

acc = forward_accumulate(df, t_base)
divisor = 1.0 if unit.startswith("Degree-hours") else 24.0
unit_label = "ADH" if divisor == 1.0 else "ADD"
target_display = target_adh / divisor

reached_idx = acc.index[acc["ADH_cumulative"] >= target_adh]
reached = len(reached_idx) > 0
reach_time = acc.loc[reached_idx[0], "Date/Time"] if reached else None

# ---------------------------------------------------------------- headline
c1, c2, c3 = st.columns(3)
c1.metric(f"Banked over the whole record ({unit_label})",
          f"{acc['ADH_cumulative'].iloc[-1] / divisor:,.0f}")
c2.metric(f"Requirement ({unit_label})", f"{target_display:,.0f}")
if reached:
    elapsed = (reach_time - acc["Date/Time"].iloc[0]).total_seconds() / 86400
    c3.metric("Requirement met", reach_time.strftime("%d %b %H:%M"),
              f"{elapsed:.2f} days in")
else:
    c3.metric("Requirement met", "Not within this record")

if not reached:
    st.warning("The record ends before the requirement is met. That is a "
               "finding, not a failure — say so rather than extrapolating.")

# ---------------------------------------------------------------- plots
fig = go.Figure()
fig.add_trace(go.Scatter(x=acc["Date/Time"], y=acc["Temp_C"], name="Temperature (°C)",
                         line=dict(color=TEAL_M, width=1.5)))
fig.add_hline(y=t_base, line_dash="dot", line_color=RUST,
              annotation_text=f"Tbase = {t_base:g} °C", annotation_position="top left")
fig.update_layout(yaxis_title="°C", xaxis_title=None, height=280, **PLOT_LAYOUT)
st.plotly_chart(fig, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=acc["Date/Time"], y=acc["ADH_cumulative"] / divisor,
                          name=f"Accumulated {unit_label}", fill="tozeroy",
                          line=dict(color=TEAL_D, width=2)))
fig2.add_hline(y=target_display, line_dash="dash", line_color=GOLD,
               annotation_text=f"{stage} — {target_display:,.0f} {unit_label}",
               annotation_position="top left")
if reached:
    fig2.add_vline(x=reach_time, line_color=RUST, line_width=2,
                   annotation_text="requirement met", annotation_position="top right")
fig2.update_layout(yaxis_title=f"Accumulated {unit_label}", height=330, **PLOT_LAYOUT)
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------- sensitivity
st.subheader("How much does Tbase matter?")
st.caption("Same record. Same requirement. Only the assumption changes.")

rows = []
for tb in np.arange(max(4.0, t_base - 4), t_base + 4.5, 1.0):
    a = forward_accumulate(df, tb)
    hit = a.index[a["ADH_cumulative"] >= target_adh]
    if len(hit):
        when = a.loc[hit[0], "Date/Time"]
        rows.append({"Tbase (°C)": f"{tb:g}",
                     "Requirement met": when.strftime("%d %b %H:%M"),
                     "Days from record start":
                         round((when - a["Date/Time"].iloc[0]).total_seconds() / 86400, 2)})
    else:
        rows.append({"Tbase (°C)": f"{tb:g}", "Requirement met": "not within record",
                     "Days from record start": None})
sens = pd.DataFrame(rows)
st.dataframe(sens, use_container_width=True, hide_index=True)

spread = sens["Days from record start"].dropna()
if len(spread) > 1:
    st.info(f"Across this range of Tbase the answer moves by "
            f"**{spread.max() - spread.min():.2f} days** — from identical data.")

with st.expander("The interval table"):
    show = acc[["Date/Time", "Temp_C", "Above_base_C", "Hours",
                "ADH_interval", "ADH_cumulative"]].copy()
    show[["Above_base_C", "ADH_interval", "ADH_cumulative"]] = \
        show[["Above_base_C", "ADH_interval", "ADH_cumulative"]].round(1)
    st.dataframe(show, use_container_width=True, hide_index=True, height=320)

st.markdown(
    f"<div style='background:#F2F6F7;border-left:4px solid {RUST};padding:14px 18px;"
    f"margin-top:20px;font-size:1.05rem;color:{SLATE}'>"
    f"<b>The app computes. It does not know.</b><br>"
    f"<span style='color:{MUTED}'>It does not know where the temperature record "
    f"was taken, whether the requirement applies to this population, or whether "
    f"anything was developing at all.</span></div>",
    unsafe_allow_html=True)
