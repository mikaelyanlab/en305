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


# ===========================================================
# Shared helpers — inlined so this file runs on its own.
# ===========================================================
# --- course palette (matches the syllabus and slide deck) -------------------
TEAL_D = "#14606B"
TEAL_M = "#2E7C7B"
RUST = "#9C4A1A"
GOLD = "#9A7A42"
GREEN = "#5E8A63"
SLATE = "#2C3E45"
MUTED = "#7C8F96"
PAPER = "#F2F6F7"

PLOT_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Calibri, Arial, sans-serif", size=13, color=SLATE),
    margin=dict(l=60, r=30, t=40, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)


# ---------------------------------------------------------------------------
# Thermal accumulation
# ---------------------------------------------------------------------------
def interval_hours(times):
    """Hours represented by each row. Last row inherits the previous spacing."""
    t = pd.to_datetime(pd.Series(times)).reset_index(drop=True)
    dt = t.diff().dt.total_seconds() / 3600.0
    dt.iloc[0] = dt.iloc[1] if len(dt) > 1 and not pd.isna(dt.iloc[1]) else 1.0
    return dt.ffill().to_numpy()


def degree_hours(temps, hours, t_base):
    """
    Accumulated degree-hours per interval.

    Temperatures at or below t_base bank ZERO. They do not bank a negative
    value — the organism does not un-develop when it is cold. This is the
    single most consequential line in the file.
    """
    above = np.maximum(0.0, np.asarray(temps, dtype=float) - float(t_base))
    return above * np.asarray(hours, dtype=float)


def forward_accumulate(df, t_base, temp_col="Temp_C", time_col="Date/Time"):
    """Walk forwards from the first row, banking degree-hours as we go."""
    out = df.copy().sort_values(time_col).reset_index(drop=True)
    hrs = interval_hours(out[time_col])
    out["Hours"] = hrs
    out["Above_base_C"] = np.maximum(0.0, out[temp_col].astype(float) - float(t_base))
    out["ADH_interval"] = degree_hours(out[temp_col], hrs, t_base)
    out["ADH_cumulative"] = out["ADH_interval"].cumsum()
    out["ADD_cumulative"] = out["ADH_cumulative"] / 24.0
    return out


def backward_walk(df, collection_time, target_adh, t_base,
                  temp_col="Temp_C", time_col="Date/Time"):
    """
    Start at collection and walk BACKWARDS until target_adh is banked.

    Returns (table, onset_time, reached). `reached` is False when the record
    runs out before the requirement is met — which is a finding, not an error,
    and the app must say so rather than returning the earliest timestamp as if
    it were an answer.
    """
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col])
    d = d[d[time_col] <= pd.to_datetime(collection_time)]
    d = d.sort_values(time_col, ascending=False).reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(), None, False

    # Resolution-independent: pandas may hold ns or us timestamps, so never
    # convert to raw integers here.
    deltas = d[time_col].diff().dt.total_seconds().abs() / 3600.0
    hrs = np.array(deltas.shift(-1).to_numpy(), dtype=float, copy=True)
    if len(hrs) > 1 and not np.isnan(hrs[-2]):
        hrs[-1] = hrs[-2]
    else:
        hrs[-1] = 1.0
    hrs = np.nan_to_num(hrs, nan=1.0)

    rows, running = [], 0.0
    onset, reached = None, False
    for i in range(len(d)):
        temp = float(d.loc[i, temp_col])
        above = max(0.0, temp - float(t_base))
        step_h = float(hrs[i])
        gained = above * step_h

        if running + gained >= target_adh and above > 0:
            need = target_adh - running
            part_h = need / above
            running = target_adh
            onset = d.loc[i, time_col] - pd.Timedelta(hours=part_h)
            rows.append(dict(**{time_col: d.loc[i, time_col]}, Temp_C=temp,
                             Above_base_C=above, Hours=round(part_h, 2),
                             ADH_interval=round(need, 1),
                             ADH_running=round(running, 1)))
            reached = True
            break

        running += gained
        rows.append(dict(**{time_col: d.loc[i, time_col]}, Temp_C=temp,
                         Above_base_C=above, Hours=step_h,
                         ADH_interval=round(gained, 1),
                         ADH_running=round(running, 1)))

    return pd.DataFrame(rows), onset, reached


# ---------------------------------------------------------------------------
# Sample data — replace with real microcosm records
# ---------------------------------------------------------------------------
def make_temperature_record(days=8.0, start="2026-09-01 05:00", step_min=60,
                            mean_c=20.0, amplitude=4.5, seed=305,
                            cold_night=None):
    """
    Diurnal temperature record. `cold_night` = (day_offset, drop_C) inserts a
    cold stretch, which is how a walk gets longer without banking anything.
    """
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 60 / step_min)
    t = pd.date_range(start=start, periods=n, freq=f"{step_min}min")
    hours = (t - t[0]).total_seconds() / 3600.0
    temp = mean_c + amplitude * np.sin(2 * np.pi * (hours - 9) / 24.0)
    temp += rng.normal(0, 0.45, n)
    temp += np.linspace(0, -2.5, n)  # gentle seasonal cooling
    if cold_night:
        off, drop = cold_night
        mask = (hours >= off * 24) & (hours < off * 24 + 12)
        temp[mask] -= drop
    return pd.DataFrame({"Date/Time": t, "Temp_C": np.round(temp, 2)})


def make_ammonia_thermal_record(days=15.0, start="2026-09-01 00:00", step_min=30,
                                seed=305):
    """Ammonia and thermal min/mean/max, shaped like the rig's output."""
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 60 / step_min)
    t = pd.date_range(start=start, periods=n, freq=f"{step_min}min")
    d = (t - t[0]).total_seconds() / 86400.0

    ammonia = (6.2 * np.exp(-((d - 4.2) ** 2) / 5.0)
               + 2.1 * np.exp(-((d - 11.5) ** 2) / 6.0)
               + rng.normal(0, 0.18, n)).clip(min=0)

    diurnal = 3.6 * np.sin(2 * np.pi * (d - 0.35))
    mass_heat = 2.9 * np.exp(-((d - 4.0) ** 2) / 1.6)  # feeding aggregation
    mean = 22.5 + diurnal * 0.55 + mass_heat + rng.normal(0, 0.25, n)
    spread = 1.4 + 2.2 * np.exp(-((d - 4.0) ** 2) / 2.2) + rng.normal(0, 0.12, n)

    return pd.DataFrame({
        "Date/Time": t,
        "Ammonia_ppm": np.round(ammonia, 3),
        "Thermal_min_C": np.round(mean - spread.clip(min=0.2), 2),
        "Thermal_mean_C": np.round(mean, 2),
        "Thermal_max_C": np.round(mean + spread.clip(min=0.2), 2),
    })


DEV_REFERENCES = {
    "L1 (first instar)": 120,
    "L2 (second instar)": 300,
    "L3 (third instar, feeding)": 600,
    "L3 (post-feeding / wandering)": 850,
    "Pupariation": 1200,
    "Adult eclosion": 2400,
}


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
