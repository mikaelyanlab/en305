# W1 — Observation vs. Inference App
# ENT 305: Forensic Entomology and the Ecology of Decay
#
# Purpose:
# Students decide whether each statement is an observation, inference, or unsupported claim.
# The key move is not the label itself; it is whether they can anchor the statement to evidence.
#
# Run with:
# streamlit run w1_observation_vs_inference.py

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Observation vs. Inference",
    page_icon="🪰",
    layout="wide",
)

LABELS = ["Observation", "Inference", "Unsupported claim"]

REASON_OPTIONS = [
    "It is directly stated in the scene.",
    "It is a reasonable interpretation of visible evidence.",
    "It requires information not provided in the scene.",
    "It treats absence from the photograph as absence over time.",
    "It makes a precise timing claim without developmental evidence.",
    "It turns a possible effect into a certain conclusion.",
]

CASES = {
    "case1": {
        "title": "Case 1 — Larvae on remains",
        "scene": """
A small decomposing animal is found on a shaded porch. In the photograph, several pale larvae are visible near the mouth opening. The body is partly covered by a cloth. No adult flies are visible in the photograph. The air temperature at discovery is 26°C.
        """.strip(),
        "evidence_bits": [],
        "claims": [
            {
                "text": "Larvae are visible near the mouth opening.",
                "answer": "Observation",
                "reason": "It is directly stated in the scene.",
                "feedback": "This is directly stated in the scene. No interpretation is needed.",
            },
            {
                "text": "Insects had access to the remains before discovery.",
                "answer": "Inference",
                "reason": "It is a reasonable interpretation of visible evidence.",
                "feedback": "Larvae support insect access at some point, but this is still an interpretation rather than a direct observation.",
            },
            {
                "text": "The animal died exactly two days before discovery.",
                "answer": "Unsupported claim",
                "reason": "It makes a precise timing claim without developmental evidence.",
                "feedback": "The scene shows larvae, but it does not provide species, developmental stage, larval age, or temperature history needed for a time estimate.",
            },
            {
                "text": "The cloth may have affected insect access to the body.",
                "answer": "Inference",
                "reason": "It is a reasonable interpretation of visible evidence.",
                "feedback": "The cloth makes this a reasonable possibility, but the scene does not prove how much access was affected.",
            },
            {
                "text": "No adult flies ever visited the remains.",
                "answer": "Unsupported claim",
                "reason": "It treats absence from the photograph as absence over time.",
                "feedback": "The photograph only says adult flies are not visible at that moment. It does not tell us what happened earlier.",
            },
            {
                "text": "The remains were found in a shaded location.",
                "answer": "Observation",
                "reason": "It is directly stated in the scene.",
                "feedback": "This is directly stated in the scene description.",
            },
        ],
    },
    "case2": {
        "title": "Case 2 — Puparia near a window",
        "scene": """
A decomposing body is discovered indoors. Several empty puparia are found along the baseboard near a closed window. No larvae are visible on the body in the photograph. The room is described as warm.
        """.strip(),
        "evidence_bits": [],
        "claims": [
            {
                "text": "Empty puparia are present near the window.",
                "answer": "Observation",
                "reason": "It is directly stated in the scene.",
                "feedback": "This is directly stated in the scene.",
            },
            {
                "text": "Some insects completed development before discovery.",
                "answer": "Inference",
                "reason": "It is a reasonable interpretation of visible evidence.",
                "feedback": "Empty puparia support the inference that immature development had already reached pupation and adult emergence.",
            },
            {
                "text": "The body must have been in this room since death.",
                "answer": "Unsupported claim",
                "reason": "It requires information not provided in the scene.",
                "feedback": "Puparia in the room do not prove the entire location history of the body.",
            },
            {
                "text": "Larvae may have moved away from the body before pupation.",
                "answer": "Inference",
                "reason": "It is a reasonable interpretation of visible evidence.",
                "feedback": "Puparia away from the body make larval migration plausible, but the route and timing are not directly observed.",
            },
            {
                "text": "The closed window proves that flies could not have entered the room.",
                "answer": "Unsupported claim",
                "reason": "It turns a possible effect into a certain conclusion.",
                "feedback": "The window was closed at discovery, but the scene does not establish whether it was always closed or whether other access routes existed.",
            },
        ],
    },
    "case3": {
        "title": "Case 3 — Wrapped remains",
        "scene": """
A decomposing body is found wrapped in a tarp in a basement. No larvae are visible in the photograph. There is strong odor and visible fluid staining beneath the body.
        """.strip(),
        "evidence_bits": [],
        "claims": [
            {
                "text": "No larvae are visible in the photograph.",
                "answer": "Observation",
                "reason": "It is directly stated in the scene.",
                "feedback": "This is a direct observation about the photograph.",
            },
            {
                "text": "No insects ever colonized the body.",
                "answer": "Unsupported claim",
                "reason": "It treats absence from the photograph as absence over time.",
                "feedback": "No visible larvae in one photograph does not prove insects were absent at all times or in all parts of the body.",
            },
            {
                "text": "The tarp may have limited insect access.",
                "answer": "Inference",
                "reason": "It is a reasonable interpretation of visible evidence.",
                "feedback": "A tarp could limit access, but the scene does not prove how complete or how long-lasting that barrier was.",
            },
            {
                "text": "The basement prevented all insect activity.",
                "answer": "Unsupported claim",
                "reason": "It turns a possible effect into a certain conclusion.",
                "feedback": "Indoor location may affect access, but the scene does not prove complete exclusion of insects.",
            },
            {
                "text": "The body shows signs of decomposition.",
                "answer": "Inference",
                "reason": "It is a reasonable interpretation of visible evidence.",
                "feedback": "Odor and fluid staining support decomposition, though they do not by themselves establish a precise stage or time since death.",
            },
        ],
    },
}

# -----------------------------
# Header
# -----------------------------

st.title("Observation vs. Inference")
st.caption("How much can you actually say from the evidence?")

st.markdown(
    "Read the scene. For each statement, decide whether it is a direct observation, a reasonable inference, "
    "or a claim that goes beyond the evidence. Then choose the reasoning rule that best explains your classification."
)

with st.expander("Definitions", expanded=True):
    st.markdown(
        "**Observation** — directly seen, measured, photographed, or stated.\n\n"
        "**Inference** — a reasonable conclusion based on evidence.\n\n"
        "**Unsupported claim** — a statement that may be possible, but is not justified by the evidence given."
    )

# -----------------------------
# Case selection
# -----------------------------

case_key = st.selectbox(
    "Choose a case",
    list(CASES.keys()),
    format_func=lambda k: CASES[k]["title"],
)
case = CASES[case_key]

st.header(case["title"])
st.info(case["scene"])

st.subheader("Classify each statement")

for i, claim in enumerate(case["claims"]):
    st.markdown(f"### Statement {i + 1}")
    st.markdown(f"**{claim['text']}**")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.radio(
            "What kind of statement is this?",
            LABELS,
            key=f"{case_key}_label_{i}",
            horizontal=True,
        )

    with col2:
        st.selectbox(
            "What makes this statement defensible — or not?",
            REASON_OPTIONS,
            key=f"{case_key}_reason_{i}",
        )

    st.text_area(
        "Explain your choice in one sentence.",
        key=f"{case_key}_explain_{i}",
        height=80,
        placeholder="Because...",
    )

    st.divider()

# -----------------------------
# Submission and feedback
# -----------------------------

if st.button("Check my work", type="primary"):
    rows = []
    label_correct = 0
    anchor_correct = 0

    for i, claim in enumerate(case["claims"]):
        student_label = st.session_state[f"{case_key}_label_{i}"]
        student_reason = st.session_state[f"{case_key}_reason_{i}"]
        explanation = st.session_state[f"{case_key}_explain_{i}"]

        label_ok = student_label == claim["answer"]
        reason_ok = student_reason == claim["reason"]

        label_correct += int(label_ok)
        anchor_correct += int(reason_ok)

        rows.append(
            {
                "Statement": claim["text"],
                "Your label": student_label,
                "Best label": claim["answer"],
                "Label match": "Yes" if label_ok else "No",
                "Your reasoning rule": student_reason,
                "Best reasoning rule": claim["reason"],
                "Reason match": "Yes" if reason_ok else "No",
                "Your explanation": explanation,
                "Feedback": claim["feedback"],
            }
        )

    result_df = pd.DataFrame(rows)
    n = len(case["claims"])
    label_score = label_correct / n
    anchor_score = anchor_correct / n
    total_score = 0.5 * label_score + 0.5 * anchor_score

    st.header("Feedback")

    c1, c2, c3 = st.columns(3)
    c1.metric("Classification", f"{label_correct}/{n}")
    c2.metric("Reasoning rule", f"{anchor_correct}/{n}")
    c3.metric("Overall", f"{total_score * 100:.0f}%")

    st.dataframe(result_df, hide_index=True, use_container_width=True)

    st.subheader("Final reflection")
    st.text_area(
        "Choose one unsupported claim from the table. What specific evidence would you need before that claim could become defensible?",
        height=120,
        key=f"{case_key}_final_reflection",
    )

    export_df = result_df.copy()
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download feedback as CSV",
        data=csv,
        file_name=f"{case_key}_observation_inference_feedback.csv",
        mime="text/csv",
    )
