# W1 — Observation vs. Inference App
# ENT 305: Forensic Entomology and the Ecology of Decay
# Streamlit app for training students to distinguish observations, defensible inferences,
# and unsupported claims from forensic entomology evidence.

import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="W1 Evidence Ladder",
    page_icon="🪰",
    layout="wide",
)

# -----------------------------
# Data model
# -----------------------------

@dataclass
class Statement:
    text: str
    correct_label: str
    defensibility: str
    explanation: str


@dataclass
class Case:
    case_id: str
    title: str
    short_title: str
    scenario: str
    known_evidence: List[str]
    unknowns: List[str]
    statements: List[Statement]


LABELS = ["Observation", "Inference", "Unsupported claim"]

CASES: Dict[str, Case] = {
    "case_1": Case(
        case_id="case_1",
        title="Case 1: Porch Discovery",
        short_title="Larvae on remains",
        scenario=(
            "A deceased small mammal is found on a shaded porch. Several pale larvae are visible "
            "in the mouth opening. No adult flies are visible in the photograph. The body is partially "
            "covered by a cloth. Ambient temperature at the time of discovery is 26°C."
        ),
        known_evidence=[
            "Several pale larvae are visible in the mouth opening.",
            "The remains are on a shaded porch.",
            "The body is partially covered by a cloth.",
            "No adult flies are visible in the photograph.",
            "Ambient temperature at discovery is 26°C.",
        ],
        unknowns=[
            "Species identity of the larvae.",
            "Developmental stage or larval age.",
            "Full temperature history before discovery.",
            "Whether colonization was delayed by the cloth.",
            "Whether the remains were moved before discovery.",
        ],
        statements=[
            Statement(
                "Larvae are visible in the mouth opening.",
                "Observation",
                "Strong",
                "This is directly stated in the evidence description.",
            ),
            Statement(
                "The remains were colonized by insects before discovery.",
                "Inference",
                "Strong",
                "Larvae on the remains support prior insect activity, although the exact timing is not established.",
            ),
            Statement(
                "Adult flies were present at the time the photograph was taken.",
                "Unsupported claim",
                "Weak",
                "The evidence says no adult flies are visible. Adult flies may have been present earlier, but that is not shown.",
            ),
            Statement(
                "The cloth may have affected access by insects.",
                "Inference",
                "Moderate",
                "A partial covering could alter access, but the app evidence does not prove the magnitude or timing of that effect.",
            ),
            Statement(
                "The animal died exactly 48 hours before discovery.",
                "Unsupported claim",
                "Weak",
                "An exact PMI requires much more evidence, including species identity, developmental stage, and temperature history.",
            ),
            Statement(
                "The larvae indicate that the remains were available to ovipositing insects at some point.",
                "Inference",
                "Strong",
                "Larvae are consistent with earlier oviposition or larviposition, but this does not establish exact time of death.",
            ),
            Statement(
                "The shaded porch caused slower larval development than a sun-exposed location would have.",
                "Unsupported claim",
                "Weak",
                "Shade could affect temperature, but the comparison requires actual thermal history or a matched sun-exposed condition.",
            ),
            Statement(
                "Additional temperature history would be needed before estimating larval age confidently.",
                "Inference",
                "Strong",
                "Development is temperature-dependent, so a single discovery temperature is not enough for a strong developmental estimate.",
            ),
        ],
    ),
    "case_2": Case(
        case_id="case_2",
        title="Case 2: Puparia Near a Window",
        short_title="Pupae near a window",
        scenario=(
            "A decomposing body is discovered indoors. Several empty puparia are found along the baseboard "
            "near a closed window. No larvae are visible on the body in the photograph. The room is warm, "
            "but no continuous temperature record is available."
        ),
        known_evidence=[
            "Several empty puparia are present near the baseboard by a window.",
            "The window is closed at discovery.",
            "No larvae are visible on the body in the photograph.",
            "The room is described as warm.",
            "No continuous temperature record is available.",
        ],
        unknowns=[
            "Whether the window was closed during the entire decomposition period.",
            "Species identity of the insects represented by the puparia.",
            "Whether larvae migrated from the body before pupation.",
            "Whether the body was moved.",
            "Temperature history of the room.",
        ],
        statements=[
            Statement(
                "Empty puparia are present near the window.",
                "Observation",
                "Strong",
                "This is directly provided in the scene description.",
            ),
            Statement(
                "Larvae may have migrated away from the body before pupation.",
                "Inference",
                "Moderate",
                "Many fly larvae leave a food source to pupate, so this is plausible, but not proven without species and developmental evidence.",
            ),
            Statement(
                "The victim definitely died in this room.",
                "Unsupported claim",
                "Weak",
                "Puparia in the room do not rule out body movement or later relocation.",
            ),
            Statement(
                "The insects may have moved toward the window or room edge before pupation.",
                "Inference",
                "Moderate",
                "The location of puparia supports a movement hypothesis, but the direction, motivation, and timing remain uncertain.",
            ),
            Statement(
                "The window was closed throughout the entire period of decomposition.",
                "Unsupported claim",
                "Weak",
                "The evidence only says the window was closed at discovery.",
            ),
            Statement(
                "The absence of visible larvae on the body proves insects did not feed on the body.",
                "Unsupported claim",
                "Weak",
                "Empty puparia could indicate that larvae already completed feeding and left the body.",
            ),
            Statement(
                "The empty puparia suggest that at least some insects completed larval development before discovery.",
                "Inference",
                "Strong",
                "Empty puparia support the conclusion that insects reached and exited the pupal stage, though species and timing remain unknown.",
            ),
            Statement(
                "A continuous temperature record would strengthen any developmental timeline.",
                "Inference",
                "Strong",
                "Developmental estimates depend on accumulated temperature, not just a general description of the room as warm.",
            ),
        ],
    ),
    "case_3": Case(
        case_id="case_3",
        title="Case 3: Wrapped Remains in a Basement",
        short_title="No insects visible",
        scenario=(
            "A decomposing body is found wrapped in a tarp in a basement. No larvae are visible in the photograph. "
            "The body has strong odor and visible fluid staining beneath it. The basement door was reportedly closed, "
            "but the reliability of that history is unknown."
        ),
        known_evidence=[
            "The body is wrapped in a tarp.",
            "The body is located in a basement.",
            "No larvae are visible in the photograph.",
            "Strong odor is reported.",
            "Visible fluid staining is present beneath the body.",
        ],
        unknowns=[
            "Whether insects had any access before discovery.",
            "Whether insects are present but hidden by the tarp or not visible in the photograph.",
            "How long the body was wrapped.",
            "Whether the body was moved from another location.",
            "Temperature and humidity history of the basement.",
        ],
        statements=[
            Statement(
                "No larvae are visible in the photograph.",
                "Observation",
                "Strong",
                "This is directly stated in the evidence description.",
            ),
            Statement(
                "The tarp may have delayed or limited insect access.",
                "Inference",
                "Moderate",
                "A wrapping can limit access, but the evidence does not prove how much delay occurred.",
            ),
            Statement(
                "No insects ever colonized the body.",
                "Unsupported claim",
                "Weak",
                "The photograph does not establish complete absence of insects over the entire postmortem period.",
            ),
            Statement(
                "The body is decomposing.",
                "Inference",
                "Strong",
                "Odor and fluid staining strongly support decomposition, although the app description does not provide a direct chemical measurement.",
            ),
            Statement(
                "The basement prevented all insect activity.",
                "Unsupported claim",
                "Weak",
                "Basement location may reduce access, but it does not prove complete prevention.",
            ),
            Statement(
                "The lack of visible larvae makes an insect-based PMI estimate more uncertain.",
                "Inference",
                "Strong",
                "Without visible insects or collected specimens, insect-based estimation has less direct support.",
            ),
            Statement(
                "The body must have been wrapped immediately after death.",
                "Unsupported claim",
                "Weak",
                "The evidence does not establish when the tarp was applied.",
            ),
            Statement(
                "Additional inspection under and inside the tarp would be needed before concluding insects are absent.",
                "Inference",
                "Strong",
                "The visible photograph may miss hidden larvae, eggs, puparia, or adult insects.",
            ),
        ],
    ),
}

# -----------------------------
# Helper functions
# -----------------------------

def initialize_state():
    if "submitted_cases" not in st.session_state:
        st.session_state.submitted_cases = {}
    if "written_answers" not in st.session_state:
        st.session_state.written_answers = {}


def label_badge(label: str) -> str:
    colors = {
        "Observation": "#d8f3dc",
        "Inference": "#fff3bf",
        "Unsupported claim": "#ffd6d6",
    }
    color = colors.get(label, "#eeeeee")
    return f"<span style='background-color:{color}; padding:0.25rem 0.5rem; border-radius:0.5rem; font-weight:600;'>{label}</span>"


def calculate_case_results(case: Case) -> pd.DataFrame:
    rows = []
    correct_count = 0

    for i, statement in enumerate(case.statements):
        key = f"{case.case_id}_statement_{i}"
        student_label = st.session_state.get(key, "")
        is_correct = student_label == statement.correct_label
        if is_correct:
            correct_count += 1
        rows.append(
            {
                "Statement": statement.text,
                "Your classification": student_label,
                "Recommended classification": statement.correct_label,
                "Aligned?": "Yes" if is_correct else "No",
                "Defensibility": statement.defensibility,
                "Why": statement.explanation,
            }
        )

    return pd.DataFrame(rows)


def case_score(case: Case) -> float:
    results = calculate_case_results(case)
    if len(results) == 0:
        return 0.0
    return float((results["Aligned?"] == "Yes").mean())


def sorting_points_from_accuracy(accuracy: float) -> int:
    if accuracy >= 0.80:
        return 4
    if accuracy >= 0.60:
        return 3
    if accuracy >= 0.40:
        return 2
    if accuracy > 0:
        return 1
    return 0


def build_download_df(case: Case) -> pd.DataFrame:
    result_df = calculate_case_results(case)
    answers = st.session_state.written_answers.get(case.case_id, {})
    meta_rows = pd.DataFrame(
        [
            {"Statement": "WRITTEN RESPONSE 1: strongest defensible inference", "Your classification": answers.get("strongest", ""), "Recommended classification": "", "Aligned?": "", "Defensibility": "", "Why": ""},
            {"Statement": "WRITTEN RESPONSE 2: claim that goes beyond evidence", "Your classification": answers.get("overreach", ""), "Recommended classification": "", "Aligned?": "", "Defensibility": "", "Why": ""},
            {"Statement": "WRITTEN RESPONSE 3: additional evidence needed", "Your classification": answers.get("additional", ""), "Recommended classification": "", "Aligned?": "", "Defensibility": "", "Why": ""},
        ]
    )
    return pd.concat([result_df, meta_rows], ignore_index=True)

# -----------------------------
# Sidebar
# -----------------------------

initialize_state()

st.sidebar.title("🪰 W1 Evidence Ladder")
st.sidebar.markdown(
    "This app trains students to separate direct evidence from defensible interpretation and narrative overreach."
)

selected_case_id = st.sidebar.radio(
    "Choose a case",
    options=list(CASES.keys()),
    format_func=lambda cid: CASES[cid].short_title,
)

st.sidebar.divider()
st.sidebar.markdown("### Suggested grading")
st.sidebar.markdown(
    "- Sorting accuracy: **4 pts**\n"
    "- Strongest inference explanation: **2 pts**\n"
    "- Overextended claim explanation: **2 pts**\n"
    "- Additional evidence needed: **2 pts**"
)

st.sidebar.divider()
show_instructor_key = st.sidebar.checkbox("Show instructor key", value=False)
show_class_summary = st.sidebar.checkbox("Show class-style summary", value=False)

# -----------------------------
# Header
# -----------------------------

st.title("How Far Can the Evidence Take You?")
st.caption("W1 — Observation vs. Inference in Forensic Entomology")

with st.expander("Concept primer", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Observation")
        st.markdown("Something directly seen, measured, photographed, recorded, or explicitly stated.")
    with col2:
        st.markdown("### Inference")
        st.markdown("A defensible interpretation drawn from evidence, usually with assumptions.")
    with col3:
        st.markdown("### Unsupported claim")
        st.markdown("A claim that may sound plausible but goes beyond what the evidence currently supports.")

    st.markdown("#### Mini-example")
    mini_df = pd.DataFrame(
        {
            "Evidence statement": [
                "Maggots are present near the mouth and nose.",
                "The remains were accessible to flies at some point.",
                "The person died outdoors.",
            ],
            "Best classification": ["Observation", "Inference", "Unsupported claim"],
        }
    )
    st.dataframe(mini_df, hide_index=True, use_container_width=True)

# -----------------------------
# Main case interface
# -----------------------------

case = CASES[selected_case_id]

st.header(case.title)

left, right = st.columns([0.95, 1.25], gap="large")

with left:
    st.subheader("Evidence scenario")
    st.info(case.scenario)

    st.markdown("### Known evidence")
    for item in case.known_evidence:
        st.markdown(f"- {item}")

    st.markdown("### What you do **not** know yet")
    for item in case.unknowns:
        st.markdown(f"- {item}")

    st.markdown("### Evidence ladder")
    st.markdown(
        """
        **Directly visible or stated**  
        ↓  
        **Reasonable interpretation**  
        ↓  
        **Conditional forensic claim**  
        ↓  
        **Overextended narrative**
        """
    )

with right:
    st.subheader("Classify each statement")
    st.markdown(
        "For each card, choose the strongest classification supported by the evidence provided. "
        "Do not use outside assumptions unless the statement is explicitly framed as conditional."
    )

    for i, statement in enumerate(case.statements):
        st.markdown(f"**Statement {i + 1}.** {statement.text}")
        c1, c2 = st.columns([0.7, 0.3])
        with c1:
            st.selectbox(
                "Classification",
                LABELS,
                key=f"{case.case_id}_statement_{i}",
                label_visibility="collapsed",
            )
        with c2:
            st.slider(
                "Confidence",
                min_value=1,
                max_value=5,
                value=3,
                key=f"{case.case_id}_confidence_{i}",
                help="1 = guessing; 5 = very confident",
            )
        st.divider()

# -----------------------------
# Written critical questions
# -----------------------------

st.header("Critical questions")
st.markdown("These are the pieces I would grade, because they reveal whether students understand the evidentiary logic.")

q1 = st.text_area(
    "1. Which statement is the strongest defensible inference, and why?",
    key=f"{case.case_id}_q1",
    height=120,
    placeholder="Name the statement and explain why it is grounded in evidence but does not overclaim.",
)
q2 = st.text_area(
    "2. Which statement goes beyond the evidence?",
    key=f"{case.case_id}_q2",
    height=120,
    placeholder="Name the overextended claim and identify the missing assumption or missing evidence.",
)
q3 = st.text_area(
    "3. What additional evidence would be needed to make the unsupported claim defensible?",
    key=f"{case.case_id}_q3",
    height=120,
    placeholder="Be specific: species ID, larval stage, temperature history, access history, scene context, etc.",
)

submitted = st.button("Submit case", type="primary")

if submitted:
    st.session_state.submitted_cases[case.case_id] = True
    st.session_state.written_answers[case.case_id] = {
        "strongest": q1,
        "overreach": q2,
        "additional": q3,
    }

# -----------------------------
# Results
# -----------------------------

if st.session_state.submitted_cases.get(case.case_id, False):
    st.header("Results")

    result_df = calculate_case_results(case)
    accuracy = case_score(case)
    sorting_points = sorting_points_from_accuracy(accuracy)

    m1, m2, m3 = st.columns(3)
    m1.metric("Sorting alignment", f"{accuracy * 100:.0f}%")
    m2.metric("Sorting points", f"{sorting_points}/4")
    m3.metric("Statements", f"{len(case.statements)}")

    st.markdown("### Sorting table")
    st.dataframe(result_df, hide_index=True, use_container_width=True)

    st.markdown("### Interpretation")
    if accuracy >= 0.80:
        st.success("Strong evidence discipline: most claims were placed at the right level of support.")
    elif accuracy >= 0.60:
        st.warning("Good start, but a few claims were probably treated as more certain than the evidence allows.")
    else:
        st.error("Several claims were over- or under-classified. Revisit the difference between direct evidence and interpretation.")

    st.markdown("### Download your work")
    download_df = build_download_df(case)
    csv = download_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV for submission",
        data=csv,
        file_name=f"{case.case_id}_observation_inference_submission.csv",
        mime="text/csv",
    )

# -----------------------------
# Instructor key
# -----------------------------

if show_instructor_key:
    st.header("Instructor key")
    key_rows = []
    for statement in case.statements:
        key_rows.append(
            {
                "Statement": statement.text,
                "Correct label": statement.correct_label,
                "Defensibility": statement.defensibility,
                "Teaching note": statement.explanation,
            }
        )
    st.dataframe(pd.DataFrame(key_rows), hide_index=True, use_container_width=True)

# -----------------------------
# Class-style summary
# -----------------------------

if show_class_summary:
    st.header("Class-style summary")
    st.markdown(
        "This section currently summarizes the active student's responses. In a classroom deployment, "
        "you could connect this to a saved CSV, Google Sheet, or database to aggregate all students."
    )

    summary_rows = []
    for i, statement in enumerate(case.statements):
        student_label = st.session_state.get(f"{case.case_id}_statement_{i}", "")
        confidence = st.session_state.get(f"{case.case_id}_confidence_{i}", None)
        summary_rows.append(
            {
                "Statement": statement.text,
                "Student label": student_label,
                "Confidence": confidence,
                "Recommended label": statement.correct_label,
                "Potential discussion prompt": (
                    "Ask why this is directly visible."
                    if statement.correct_label == "Observation"
                    else "Ask what assumptions make this defensible."
                    if statement.correct_label == "Inference"
                    else "Ask what evidence would be needed before saying this."
                ),
            }
        )

    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------

st.divider()
st.markdown(
    "**Instructor note:** The goal is not to make students timid. The goal is to make them precise: "
    "insects are evidence, not a license to narrate the whole crime scene."
)
