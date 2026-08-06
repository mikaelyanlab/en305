"""
ENT 305 — Session 12 · PMI Practice Dataset
The backward clock. GRADED SESSION.

Sequence enforced by this app, per the session script:
    1. Prediction   — written and committed BEFORE the controls unlock
    2. Model        — walk the clock backwards from collection
    3. Interpretation — convert output into an interval, with a quantity name

Step 1 is graded on whether the team committed, not on whether they were right.
The gate exists so that grading claim is true.

Run:  streamlit run ent305_s12_backward_clock.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ent305_common import (DEV_REFERENCES, GOLD, GREEN, MUTED, PLOT_LAYOUT,
                           RUST, SLATE, TEAL_D, TEAL_M, backward_walk,
                           make_temperature_record)

st.set_page_config(page_title="ENT 305 · The backward clock", layout="wide")

if "committed" not in st.session_state:
    st.session_state.committed = False

# ===========================================================================
# STEP 1 — the gate
# ===========================================================================
if not st.session_state.committed:
    st.title("Before you touch the model")
    st.markdown(
        f"<div style='background:#F2F6F7;border-left:4px solid {RUST};"
        f"padding:16px 20px;font-size:1.05rem;color:{SLATE}'>"
        f"Once the app gives you a number you will find reasons that number is "
        f"sensible. Predicting first is how you catch yourself doing it."
        f"</div>", unsafe_allow_html=True)
    st.write("")

    table = st.text_input("Table number")
    st.markdown("**Which assumptions will matter most, and why?**")
    pred_assumption = st.text_area(
        "Name them before you see any output.", height=110,
        placeholder="e.g. the base temperature, because the record spends "
                    "several nights near it…")
    st.markdown("**Your predicted interval, before modelling.**")
    cols = st.columns(2)
    lo = cols[0].number_input("Shortest plausible (days)", 0.0, 60.0, 2.0, 0.25)
    hi = cols[1].number_input("Longest plausible (days)", 0.0, 60.0, 5.0, 0.25)

    ready = bool(table.strip()) and len(pred_assumption.strip()) >= 20 and hi >= lo
    if not ready:
        st.caption("Enter a table number, name at least one assumption in a "
                   "sentence, and give a range where the longest is not shorter "
                   "than the shortest.")
    if st.button("Commit and open the model", type="primary", disabled=not ready):
        st.session_state.committed = True
        st.session_state.pred = dict(table=table.strip(), why=pred_assumption.strip(),
                                     lo=lo, hi=hi)
        st.rerun()
    st.stop()

# ===========================================================================
# STEP 2 — the model
# ===========================================================================
pred = st.session_state.pred
st.title("The backward clock")
st.caption(f"Session 12 · PMI Practice Dataset — Table {pred['table']}")

with st.expander("Your committed prediction (locked)", expanded=False):
    st.write(f"**Predicted interval:** {pred['lo']:g}–{pred['hi']:g} days")
    st.write(f"**Assumptions you named:** {pred['why']}")

# ---------------------------------------------------------------- data
with st.sidebar:
    st.header("PR-1 — the record")
    src = st.radio("Temperature record", ["Built-in (PR-1)", "Upload CSV"], index=0)
    if src == "Upload CSV":
        up = st.file_uploader("Columns: Date/Time, Temp_C", type="csv")
        df = make_temperature_record(days=10) if up is None else pd.read_csv(up)
        if up is not None:
            df["Date/Time"] = pd.to_datetime(df["Date/Time"], errors="coerce")
            df = df.dropna(subset=["Date/Time"])
    else:
        df = make_temperature_record(days=10, start="2026-08-28 00:00",
                                     cold_night=(5, 6.0))

    tmin = pd.to_datetime(df["Date/Time"]).min()
    tmax = pd.to_datetime(df["Date/Time"]).max()

    st.markdown("---")
    st.header("Read from PR-1")
    coll_date = st.date_input("Collection date", value=tmax.date(),
                              min_value=tmin.date(), max_value=tmax.date())
    coll_time = st.time_input("Collection time", value=tmax.time())
    collection = pd.Timestamp.combine(pd.Timestamp(coll_date).date(), coll_time)

    stage = st.selectbox("Developmental stage reached",
                         list(DEV_REFERENCES.keys()), index=3)

    st.markdown("---")
    st.header("Your assumptions")
    st.caption("Every control below is something you are assuming, not "
               "something PR-1 measured.")
    adh_req = st.slider("Requirement for that stage (degree-hours)",
                        50, 3000, int(DEV_REFERENCES[stage]), 10,
                        help="From a published reference: another laboratory, "
                             "another diet, constant conditions.")
    t_base = st.slider("Base temperature, Tbase (°C)", 4.0, 16.0, 10.0, 0.5)
    offset = st.slider("Temperature offset (°C)", -3.0, 3.0, 0.0, 0.25,
                       help="The gap between where the logger sat and where "
                            "the specimen was. PR-1 does not tell you this.")

work = df.copy()
work["Temp_C"] = work["Temp_C"].astype(float) + offset

table_out, onset, reached = backward_walk(work, collection, adh_req, t_base)

# ---------------------------------------------------------------- interval
def onset_for(adh, tb, off):
    w = df.copy()
    w["Temp_C"] = w["Temp_C"].astype(float) + off
    _, o, r = backward_walk(w, collection, adh, tb)
    return o if r else None

lo_onset = onset_for(adh_req * 0.85, t_base + 1.0, offset + 0.75)
hi_onset = onset_for(adh_req * 1.15, t_base - 1.0, offset - 0.75)

st.subheader("Result")
if not reached:
    st.error("The record runs out before the requirement is banked. The data "
             "do not support an estimate. Saying so is the correct answer — "
             "do not extrapolate beyond the record.")
else:
    days = (collection - onset).total_seconds() / 86400
    c1, c2, c3 = st.columns(3)
    c1.metric("Development began (point estimate)", onset.strftime("%d %b %H:%M"))
    c2.metric("Before collection", f"{days:.2f} days")
    if lo_onset is not None and hi_onset is not None:
        d_lo = (collection - lo_onset).total_seconds() / 86400
        d_hi = (collection - hi_onset).total_seconds() / 86400
        c3.metric("Defensible window", f"{min(d_lo, d_hi):.2f} – {max(d_lo, d_hi):.2f} days")
    else:
        c3.metric("Defensible window", "extends beyond the record")

    st.markdown(
        f"<div style='background:#F2F6F7;border-left:4px solid {GOLD};"
        f"padding:14px 18px;color:{SLATE}'>"
        f"Say what this number <i>is</i>: the estimated point at which "
        f"<b>development began</b>. Not the moment of death. The moment the "
        f"clock started.</div>", unsafe_allow_html=True)

    pl, ph = pred["lo"], pred["hi"]
    if not (pl <= days <= ph):
        st.info(f"Your committed prediction was {pl:g}–{ph:g} days. The model "
                f"gives {days:.2f}. Do not quietly adopt the model's number — "
                f"work out which of your assumptions the difference lives in.")

# ---------------------------------------------------------------- plot
if reached and len(table_out):
    fig = go.Figure()
    d = work.copy()
    d["Date/Time"] = pd.to_datetime(d["Date/Time"])
    fig.add_trace(go.Scatter(x=d["Date/Time"], y=d["Temp_C"], name="Temperature (°C)",
                             line=dict(color=TEAL_M, width=1.4)))
    fig.add_hline(y=t_base, line_dash="dot", line_color=RUST,
                  annotation_text=f"Tbase = {t_base:g} °C")
    fig.add_vrect(x0=onset, x1=collection, fillcolor=TEAL_D, opacity=0.10,
                  line_width=0, annotation_text="banked interval",
                  annotation_position="top left")
    if lo_onset is not None and hi_onset is not None:
        fig.add_vrect(x0=min(lo_onset, hi_onset), x1=max(lo_onset, hi_onset),
                      fillcolor=GOLD, opacity=0.18, line_width=0,
                      annotation_text="window of plausible onset",
                      annotation_position="bottom left")
    fig.add_vline(x=collection, line_color=SLATE, line_width=2,
                  annotation_text="collection", annotation_position="top right")
    fig.update_layout(yaxis_title="°C", height=340, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("The walk, backwards from collection")
    st.caption("Read the top rows first. If the record is wrong at the recent "
               "end, the answer moves a lot.")
    show = table_out.rename(columns={"Above_base_C": "Above base (°C)",
                                     "ADH_interval": "Banked this step",
                                     "ADH_running": "Running total"})
    st.dataframe(show, use_container_width=True, hide_index=True, height=300)

    zero = int((table_out["Above_base_C"] <= 0).sum())
    if zero:
        st.warning(f"{zero} interval(s) sat at or below the base temperature "
                   f"and banked **zero**. Cold periods do not slow the estimate "
                   f"proportionally — they lengthen the walk without buying "
                   f"anything.")

# ---------------------------------------------------------------- sensitivity
st.subheader("Which assumption is carrying your answer?")
rows = []
for label, adh, tb, off in [
    ("As set", adh_req, t_base, offset),
    ("Tbase −1 °C", adh_req, t_base - 1, offset),
    ("Tbase +1 °C", adh_req, t_base + 1, offset),
    ("Requirement −15%", adh_req * 0.85, t_base, offset),
    ("Requirement +15%", adh_req * 1.15, t_base, offset),
    ("Offset −1 °C", adh_req, t_base, offset - 1),
    ("Offset +1 °C", adh_req, t_base, offset + 1),
]:
    o = onset_for(adh, tb, off)
    rows.append({"Change": label,
                 "Onset": o.strftime("%d %b %H:%M") if o is not None else "beyond record",
                 "Days before collection":
                     round((collection - o).total_seconds() / 86400, 2) if o is not None else None})
sens = pd.DataFrame(rows)
st.dataframe(sens, use_container_width=True, hide_index=True)

vals = sens["Days before collection"].dropna()
if len(vals) > 1:
    st.info(f"These are the same specimen and the same record. The estimate "
            f"spans **{vals.max() - vals.min():.2f} days** across assumptions "
            f"you cannot verify from PR-1.")

st.markdown(
    f"<div style='background:#F2F6F7;border-left:4px solid {RUST};padding:14px 18px;"
    f"margin-top:20px;color:{SLATE}'><b>The app is not an answer machine.</b> "
    f"It is an arithmetic machine that does not know where its inputs came from. "
    f"<span style='color:{MUTED}'>Your job is everything the app cannot do — "
    f"read the provenance notes, name the assumptions, state the window.</span>"
    f"</div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("---")
    if st.button("Reset (new team)"):
        st.session_state.committed = False
        st.rerun()
