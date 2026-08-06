"""
ENT 305 — Session 22 · Toxicology and Sensors
Ammonia and thermal record.

Rebuilt from last year's reveal dashboard, with three changes:

  1. Event markers are OFF by default. Students place the events from the data
     before seeing where they actually fall. Last year's version shipped with
     them pre-set, which handed over the answer.
  2. Nothing is silently dropped. The old version converted any thermal range
     above 15 °C to NaN and clipped both axes without saying so. A course that
     teaches students to find undocumented cleaning steps should not contain
     one. Filtering is now optional, visible, and counted.
  3. Axis limits are adjustable rather than hardcoded, and default to the data.

Run:  streamlit run ent305_s22_ammonia_thermal.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
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
    # .to_numpy() is essential: (t - t[0]).total_seconds() yields a pandas
    # Index, which is immutable, so any later in-place edit would raise.
    hours = np.asarray((t - t[0]).total_seconds() / 3600.0, dtype=float)
    temp = np.asarray(mean_c + amplitude * np.sin(2 * np.pi * (hours - 9) / 24.0),
                      dtype=float)
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
    d = np.asarray((t - t[0]).total_seconds() / 86400.0, dtype=float)

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


st.set_page_config(page_title="ENT 305 · Ammonia and thermal", layout="wide")

st.title("Ammonia and thermal record")
st.caption("Session 22 · Toxicology and Sensors — evidence streams that widen "
           "uncertainty rather than narrowing it")


def _rgba(hex_colour, alpha):
    """Plotly fills need rgba(); the palette is stored as hex."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


REQUIRED = ["Date/Time", "Ammonia_ppm", "Thermal_min_C", "Thermal_mean_C",
            "Thermal_max_C"]

# ---------------------------------------------------------------- data
with st.sidebar:
    st.header("Data")
    files = st.file_uploader("Upload one or more vessel CSVs", type="csv",
                             accept_multiple_files=True)

frames = []
if files:
    for f in files:
        d = pd.read_csv(f)
        d.columns = [c.strip().replace(" ", "_").replace("(", "")
                     .replace(")", "").replace("°C", "C") for c in d.columns]
        missing = [c for c in REQUIRED if c not in d.columns]
        if missing:
            st.warning(f"**{f.name}** is missing {missing} and was not loaded. "
                       f"Found: {list(d.columns)}")
            continue
        d["Date/Time"] = pd.to_datetime(d["Date/Time"], errors="coerce")
        for c in REQUIRED[1:]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["Vessel"] = f.name.replace(".csv", "")
        frames.append(d)
else:
    for name, seed in [("Vessel A", 305), ("Vessel B", 77)]:
        d = make_ammonia_thermal_record(seed=seed)
        d["Vessel"] = name
        frames.append(d)
    st.sidebar.info("No files uploaded — showing two built-in vessels.")

if not frames:
    st.error("No usable data.")
    st.stop()

data = pd.concat(frames, ignore_index=True)
data["Thermal_range_C"] = data["Thermal_max_C"] - data["Thermal_min_C"]

for v, g in data.groupby("Vessel"):
    t0 = g["Date/Time"].min()
    data.loc[g.index, "Elapsed_days"] = (g["Date/Time"] - t0).dt.total_seconds() / 86400

# ---------------------------------------------------------------- filtering
with st.sidebar:
    st.markdown("---")
    st.header("Filtering")
    st.caption("Off by default. Anything you exclude is counted and reported "
               "on screen — as it would have to be in a real record.")
    do_filter = st.checkbox("Exclude implausible thermal ranges", value=False)
    max_range = st.slider("Exclude range above (°C)", 2.0, 30.0, 15.0, 0.5,
                          disabled=not do_filter)

excluded = 0
if do_filter:
    mask = data["Thermal_range_C"] > max_range
    excluded = int(mask.sum())
    data.loc[mask, "Thermal_range_C"] = np.nan

missing_rows = int(data[REQUIRED[1:]].isna().any(axis=1).sum())
notes = []
if excluded:
    notes.append(f"**{excluded}** reading(s) excluded by the thermal-range "
                 f"filter at {max_range:g} °C.")
if missing_rows:
    notes.append(f"**{missing_rows}** row(s) have at least one missing value.")
if notes:
    st.warning(" ".join(notes) + " If you rely on this record, this belongs "
                                 "in your provenance notes.")
else:
    st.success("No rows excluded and no missing values. Nothing has been "
               "removed from what you are looking at.")

# ---------------------------------------------------------------- events
with st.sidebar:
    st.markdown("---")
    st.header("Event markers")
    show_events = st.checkbox("Show event markers", value=False,
                              help="Place these from the data first. Turn them "
                                   "on only when your table has committed.")
    events = {}
    if show_events:
        for label, default in [("Eggs placed", 0.0), ("L1 hatch", 1.0),
                               ("L3 feeding peak", 3.5), ("Wandering", 4.6),
                               ("Pupariation", 8.0), ("Eclosion", 14.0)]:
            events[label] = st.number_input(label, 0.0, 30.0, default, 0.1)

def add_events(fig):
    if not show_events:
        return
    for label, day in events.items():
        fig.add_vline(x=day, line_width=1, line_dash="dot", line_color=SLATE,
                      annotation_text=label, annotation_position="top left")

if not show_events:
    st.info("**Event markers are off.** Before turning them on, write down "
            "where your table thinks each event falls — then check.")

# ---------------------------------------------------------------- plot 1
st.subheader("Ammonia and thermal mean over time")
fig1 = go.Figure()
palette = [TEAL_D, RUST, GOLD, GREEN, TEAL_M]
for i, (v, g) in enumerate(data.groupby("Vessel")):
    c = palette[i % len(palette)]
    fig1.add_trace(go.Scatter(x=g["Elapsed_days"], y=g["Thermal_mean_C"],
                              name=f"{v} — thermal mean (°C)", yaxis="y1",
                              line=dict(color=c, dash="dot", width=1.3)))
    fig1.add_trace(go.Scatter(x=g["Elapsed_days"], y=g["Ammonia_ppm"],
                              name=f"{v} — ammonia (ppm)", yaxis="y2",
                              line=dict(color=c, width=2.2)))
add_events(fig1)
fig1.update_layout(xaxis_title="Elapsed days",
                   yaxis=dict(title="Thermal mean (°C)"),
                   yaxis2=dict(title="Ammonia (ppm)", overlaying="y", side="right"),
                   height=430, **PLOT_LAYOUT)
st.plotly_chart(fig1, use_container_width=True)
st.caption("Axes scale to the data. Nothing is clipped.")

# ---------------------------------------------------------------- plot 2
st.subheader("Thermal range against ammonia")
st.caption("A narrow range means a heat-buffered system. A wide one means the "
           "vessel is tracking ambient — or something inside it is generating heat.")
fig2 = go.Figure()
for i, (v, g) in enumerate(data.groupby("Vessel")):
    c = palette[i % len(palette)]
    fig2.add_trace(go.Scatter(x=g["Elapsed_days"], y=g["Thermal_range_C"],
                              name=f"{v} — thermal range (°C)", fill="tozeroy",
                              mode="none", fillcolor=_rgba(c, 0.25)))
    fig2.add_trace(go.Scatter(x=g["Elapsed_days"], y=g["Ammonia_ppm"],
                              name=f"{v} — ammonia (ppm)", yaxis="y2",
                              line=dict(color=c, width=2)))
add_events(fig2)
fig2.update_layout(xaxis_title="Elapsed days",
                   yaxis=dict(title="Thermal range (°C)"),
                   yaxis2=dict(title="Ammonia (ppm)", overlaying="y", side="right"),
                   height=400, **PLOT_LAYOUT)
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------- correlation
st.subheader("Correlation — and what it does not tell you")
rows = []
for v, g in data.groupby("Vessel"):
    rows.append({"Vessel": v,
                 "Ammonia vs thermal min": round(g["Ammonia_ppm"].corr(g["Thermal_min_C"]), 3),
                 "Ammonia vs thermal mean": round(g["Ammonia_ppm"].corr(g["Thermal_mean_C"]), 3),
                 "Ammonia vs thermal max": round(g["Ammonia_ppm"].corr(g["Thermal_max_C"]), 3),
                 "n": int(g["Ammonia_ppm"].notna().sum())})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.markdown(
    f"<div style='background:#F2F6F7;border-left:4px solid {RUST};padding:14px 18px;"
    f"color:{SLATE}'>A correlation coefficient is not a colonization time, and "
    f"neither stream carries a clock. <span style='color:{MUTED}'>Both of these "
    f"widen the interval you can defend. Neither narrows it.</span></div>",
    unsafe_allow_html=True)

with st.expander("The data"):
    st.dataframe(data, use_container_width=True, height=340)
    st.download_button("Download what is shown above (CSV)",
                       data.to_csv(index=False).encode(),
                       "ent305_session22_data.csv", "text/csv")
