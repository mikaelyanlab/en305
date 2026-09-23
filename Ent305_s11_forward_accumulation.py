"""
ENT 305 — Session 12 · Developmental Clock: Working Backward

Students receive:
    - specimen identity and developmental stage
    - a temperature record
    - an assigned developmental reference

They determine the appropriate lower developmental threshold and thermal
requirement from the reference, then enter those values here.

The app does one thing:
    Start at collection and walk backward through the temperature history
    until the required accumulated degree-hours have been reached.

THE APP COMPUTES. IT DOES NOT KNOW.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ===========================================================
# Course styling
# ===========================================================

TEAL_D = "#14606B"
TEAL_M = "#2E7C7B"
RUST = "#9C4A1A"
GOLD = "#9A7A42"
SLATE = "#2C3E45"
MUTED = "#7C8F96"
PAPER = "#F2F6F7"

PLOT_LAYOUT = dict(
    template="plotly_white",
    font=dict(
        family="Calibri, Arial, sans-serif",
        size=13,
        color=SLATE
    ),
    margin=dict(l=60, r=30, t=40, b=50),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        x=0
    ),
)


# ===========================================================
# Backward thermal-time calculation
# ===========================================================

def backward_walk(
    df,
    collection_time,
    target_adh,
    t_base,
    temp_col="Temp_C",
    time_col="Date/Time"
):
    """
    Walk backward from collection through a temperature record until
    target_adh has been accumulated.

    INTERVAL CONVENTION
    -------------------
    A temperature recorded at time t is treated as representing the
    interval from t until the next recorded timestamp.

    Example:
        13:00   20 C
        14:00   21 C

    The 20 C observation represents 13:00–14:00.

    If collection occurs between two timestamps, only the portion of
    that interval before collection is used.

    Temperatures at or below Tbase contribute zero ADH.

    Returns
    -------
    walk : DataFrame
        Intervals used while walking backward.

    onset_time : Timestamp or None
        Estimated time at which the target thermal requirement is met.

    reached : bool
        True if the available temperature record contains enough
        thermal accumulation to reach target_adh.
    """

    d = df[[time_col, temp_col]].copy()

    d[time_col] = pd.to_datetime(
        d[time_col],
        errors="coerce"
    )

    d[temp_col] = pd.to_numeric(
        d[temp_col],
        errors="coerce"
    )

    d = (
        d.dropna(subset=[time_col, temp_col])
        .sort_values(time_col)
        .drop_duplicates(subset=[time_col], keep="last")
        .reset_index(drop=True)
    )

    collection_time = pd.to_datetime(collection_time)

    # Only observations beginning before collection can contribute.
    d = d[d[time_col] < collection_time].copy()

    if d.empty:
        return pd.DataFrame(), None, False

    # Each observation represents the interval beginning at that timestamp.
    d["Interval_end"] = d[time_col].shift(-1)

    # For the final available observation, the interval ends at collection.
    d.loc[d.index[-1], "Interval_end"] = collection_time

    # If collection falls inside an existing interval, truncate at collection.
    d["Interval_end"] = d["Interval_end"].where(
        d["Interval_end"] <= collection_time,
        collection_time
    )

    d["Hours"] = (
        d["Interval_end"] - d[time_col]
    ).dt.total_seconds() / 3600.0

    # Remove zero/negative intervals.
    d = d[d["Hours"] > 0].copy()

    if d.empty:
        return pd.DataFrame(), None, False

    # Effective temperature above the developmental threshold.
    d["Above_base_C"] = np.maximum(
        0.0,
        d[temp_col].astype(float) - float(t_base)
    )

    d["ADH_interval"] = (
        d["Above_base_C"] * d["Hours"]
    )

    # Walk backward from collection.
    d = d.sort_values(
        time_col,
        ascending=False
    ).reset_index(drop=True)

    rows = []
    running = 0.0
    onset_time = None
    reached = False

    for _, row in d.iterrows():

        interval_start = row[time_col]
        interval_end = row["Interval_end"]

        temp = float(row[temp_col])
        above = float(row["Above_base_C"])
        hours = float(row["Hours"])
        available_adh = float(row["ADH_interval"])

        # Does the target fall somewhere inside this interval?
        if (
            above > 0
            and running + available_adh >= target_adh
        ):
            adh_needed = target_adh - running
            hours_needed = adh_needed / above

            # We are walking backward from interval_end.
            onset_time = (
                interval_end
                - pd.Timedelta(hours=hours_needed)
            )

            rows.append({
                "Interval_start": onset_time,
                "Interval_end": interval_end,
                "Temp_C": temp,
                "Above_base_C": above,
                "Hours": hours_needed,
                "ADH_interval": adh_needed,
                "ADH_running": target_adh,
            })

            running = target_adh
            reached = True
            break

        running += available_adh

        rows.append({
            "Interval_start": interval_start,
            "Interval_end": interval_end,
            "Temp_C": temp,
            "Above_base_C": above,
            "Hours": hours,
            "ADH_interval": available_adh,
            "ADH_running": running,
        })

    walk = pd.DataFrame(rows)

    if not walk.empty:
        numeric_cols = [
            "Temp_C",
            "Above_base_C",
            "Hours",
            "ADH_interval",
            "ADH_running",
        ]

        walk[numeric_cols] = walk[numeric_cols].round(2)

    return walk, onset_time, reached


# ===========================================================
# Streamlit page
# ===========================================================

st.set_page_config(
    page_title="ENT 305 · Developmental Clock",
    layout="wide"
)

st.title("Developmental Clock: Working Backward")

st.caption(
    "Start at collection and work backward through the temperature history."
)


# ===========================================================
# Sidebar
# ===========================================================

with st.sidebar:

    st.header("Temperature record")

    uploaded_file = st.file_uploader(
        "Upload temperature record",
        type="csv",
        help="CSV must contain Date/Time and Temp_C columns."
    )

    if uploaded_file is None:
        st.info(
            "Upload the temperature record provided for your case."
        )
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(
            f"The CSV could not be read: {exc}"
        )
        st.stop()

    required_columns = {
        "Date/Time",
        "Temp_C"
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        st.error(
            "The temperature file is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )
        st.stop()

    df["Date/Time"] = pd.to_datetime(
        df["Date/Time"],
        errors="coerce"
    )

    df["Temp_C"] = pd.to_numeric(
        df["Temp_C"],
        errors="coerce"
    )

    bad_rows = (
        df["Date/Time"].isna()
        | df["Temp_C"].isna()
    )

    n_bad = int(bad_rows.sum())

    if n_bad:
        st.warning(
            f"{n_bad} row(s) contained an unreadable "
            "date/time or temperature and were excluded."
        )

    df = (
        df.loc[~bad_rows]
        .sort_values("Date/Time")
        .drop_duplicates(
            subset=["Date/Time"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    if len(df) < 2:
        st.error(
            "The temperature record must contain at least "
            "two usable observations."
        )
        st.stop()

    st.markdown("---")

    st.header("Case inputs")

    collection_time = st.datetime_input(
        "Collection date and time",
        value=df["Date/Time"].max().to_pydatetime()
    )

    t_base = st.number_input(
        "Lower developmental threshold (°C)",
        min_value=0.0,
        max_value=30.0,
        value=None,
        step=0.1,
        placeholder="Enter from developmental reference",
        help=(
            "Enter the lower developmental threshold supported "
            "by your assigned developmental reference."
        )
    )

    target_adh = st.number_input(
        "Thermal requirement (ADH)",
        min_value=0.1,
        value=None,
        step=1.0,
        placeholder="Enter from developmental reference",
        help=(
            "Enter the thermal requirement supported by your "
            "assigned developmental reference."
        )
    )

    run = st.button(
        "Run backward",
        type="primary",
        use_container_width=True
    )


# ===========================================================
# Waiting state
# ===========================================================

if not run:

    st.info(
        "Enter the values from your developmental reference, "
        "then run the developmental clock backward."
    )

    st.stop()


# ===========================================================
# Validate case inputs
# ===========================================================

if t_base is None or target_adh is None:

    st.error(
        "Enter both the lower developmental threshold "
        "and the thermal requirement."
    )

    st.stop()


collection_time = pd.to_datetime(
    collection_time
)


if collection_time <= df["Date/Time"].min():

    st.error(
        "The collection time must occur after the beginning "
        "of the temperature record."
    )

    st.stop()


if collection_time > df["Date/Time"].max():

    st.warning(
        "The collection time occurs after the final temperature "
        "observation. The final recorded temperature will be treated "
        "as representing the interval from that observation until "
        "collection."
    )


# ===========================================================
# Run calculation
# ===========================================================

walk, onset_time, reached = backward_walk(
    df=df,
    collection_time=collection_time,
    target_adh=float(target_adh),
    t_base=float(t_base)
)


# ===========================================================
# Main results
# ===========================================================

c1, c2, c3 = st.columns(3)

c1.metric(
    "Thermal requirement",
    f"{target_adh:,.0f} ADH"
)

c2.metric(
    "Collection",
    collection_time.strftime(
        "%d %b %Y · %H:%M"
    )
)


if reached:

    elapsed_hours = (
        collection_time - onset_time
    ).total_seconds() / 3600.0

    c3.metric(
        "Developmental interval",
        f"{elapsed_hours:.1f} hours"
    )

    st.success(
        "Estimated developmental onset: "
        f"**{onset_time.strftime('%d %b %Y · %H:%M')}**"
    )

else:

    c3.metric(
        "Developmental interval",
        "Not determined"
    )

    st.warning(
        "The available temperature record ends before the "
        "required ADH is reached. A developmental-onset "
        "estimate cannot be calculated from this record."
    )


# ===========================================================
# Temperature-history plot
# ===========================================================

plot_df = df[
    df["Date/Time"] <= collection_time
].copy()


fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=plot_df["Date/Time"],
        y=plot_df["Temp_C"],
        name="Temperature",
        mode="lines",
        line=dict(
            color=TEAL_M,
            width=1.8
        )
    )
)


fig.add_hline(
    y=t_base,
    line_dash="dot",
    line_color=RUST,
    annotation_text=(
        f"Tbase = {t_base:g} °C"
    ),
    annotation_position="top left"
)


fig.add_vline(
    x=collection_time,
    line_color=SLATE,
    line_width=2,
    annotation_text="Collection",
    annotation_position="top right"
)


if reached:

    fig.add_vline(
        x=onset_time,
        line_color=RUST,
        line_width=2,
        annotation_text=(
            "Estimated developmental onset"
        ),
        annotation_position="top left"
    )

    fig.add_vrect(
        x0=onset_time,
        x1=collection_time,
        fillcolor=GOLD,
        opacity=0.12,
        line_width=0
    )


fig.update_layout(
    yaxis_title="Temperature (°C)",
    xaxis_title=None,
    height=340,
    **PLOT_LAYOUT
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ===========================================================
# Backward accumulation plot
# ===========================================================

if not walk.empty:

    # The calculation was performed backward.
    # Reconstruct cumulative ADH in chronological order so the graph
    # reads naturally from estimated onset toward collection.

    chronological = (
        walk.sort_values("Interval_start")
        .reset_index(drop=True)
        .copy()
    )

    chronological["ADH_from_onset"] = (
        chronological["ADH_interval"]
        .cumsum()
    )

    # Build a step-like series with a true zero at onset.
    x_values = []
    y_values = []

    if reached:
        x_values.append(onset_time)
        y_values.append(0.0)
    else:
        x_values.append(
            chronological.loc[
                0,
                "Interval_start"
            ]
        )
        y_values.append(0.0)

    running_forward = 0.0

    for _, row in chronological.iterrows():

        running_forward += float(
            row["ADH_interval"]
        )

        x_values.append(
            row["Interval_end"]
        )

        y_values.append(
            running_forward
        )


    fig2 = go.Figure()


    fig2.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            name="Accumulated ADH",
            mode="lines",
            line=dict(
                color=TEAL_D,
                width=2
            ),
            fill="tozeroy"
        )
    )


    fig2.add_hline(
        y=target_adh,
        line_dash="dash",
        line_color=GOLD,
        annotation_text=(
            f"Requirement = "
            f"{target_adh:,.0f} ADH"
        ),
        annotation_position="top left"
    )
# Label the estimated beginning of development
if reached:
    fig2.add_annotation(
        x=onset_time,
        y=0,
        text=(
            "<b>Estimated developmental onset</b><br>"
            f"{onset_time.strftime('%b %d, %Y · %H:%M')}"
        ),
        showarrow=True,
        arrowhead=2,
        arrowcolor=RUST,
        ax=60,
        ay=-55,
        font=dict(
            color=RUST,
            size=12
        ),
        bgcolor="white",
        bordercolor=RUST,
        borderwidth=1
    )

# Label collection
fig2.add_annotation(
    x=collection_time,
    y=target_adh,
    text=(
        "<b>Collection</b><br>"
        f"{collection_time.strftime('%b %d, %Y · %H:%M')}"
    ),
    showarrow=True,
    arrowhead=2,
    arrowcolor=SLATE,
    ax=-70,
    ay=55,
    font=dict(
        color=SLATE,
        size=12
    ),
    bgcolor="white",
    bordercolor=SLATE,
    borderwidth=1
)


    fig2.update_layout(
        yaxis_title=(
            "Accumulated degree-hours (ADH)"
        ),
        xaxis_title=None,
        height=330,
        **PLOT_LAYOUT
    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ===========================================================
# Audit table
# ===========================================================

with st.expander(
    "Show the calculation"
):

    if walk.empty:

        st.write(
            "No usable intervals were available."
        )

    else:

        show = walk[
            [
                "Interval_start",
                "Interval_end",
                "Temp_C",
                "Above_base_C",
                "Hours",
                "ADH_interval",
                "ADH_running",
            ]
        ].copy()


        show = show.rename(
            columns={
                "Interval_start":
                    "Interval begins",

                "Interval_end":
                    "Interval ends",

                "Temp_C":
                    "Temperature (°C)",

                "Above_base_C":
                    "Degrees above Tbase",

                "Hours":
                    "Hours used",

                "ADH_interval":
                    "ADH gained",

                "ADH_running":
                    "ADH accumulated backward",
            }
        )


        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            height=340
        )


# ===========================================================
# Footer
# ===========================================================

st.markdown(
    f"""
    <div style="
        background:{PAPER};
        border-left:4px solid {RUST};
        padding:14px 18px;
        margin-top:20px;
        font-size:1.05rem;
        color:{SLATE};
    ">
        <b>THE APP COMPUTES. IT DOES NOT KNOW.</b><br>
        <span style="color:{MUTED}">
        You supplied the temperature record, developmental threshold,
        thermal requirement, and collection time. The calculation cannot
        determine whether those inputs are appropriate.
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
