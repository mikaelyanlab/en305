# W1 — Observation vs. Inference App
# ENT 305: Forensic Entomology and the Ecology of Decay
#
# This version is designed as a reasoning simulator, not an interactive worksheet.
# Students begin with incomplete evidence, classify claims, choose which evidence to request,
# revise their classifications, and see how claim defensibility changes as evidence accumulates.
#
# Core lesson:
# Insects are evidence. They are not permission to narrate the whole crime scene.

import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

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

LABELS = ["Observation", "Inference", "Unsupported claim"]
CONFIDENCE_LABELS = {
    1: "1 — guessing",
    2: "2 — unsure",
    3: "3 — moderately sure",
    4: "4 — confident",
    5: "5 — very confident",
}

EVIDENCE_TAG_DESCRIPTIONS = {
    "visible_larvae": "larvae are visible in the scene",
    "visible_puparia": "puparia are visible in the scene",
    "species_id": "species identity has been established",
    "developmental_stage": "developmental stage has been established",
    "larval_measurement": "larvae have been measured",
    "temperature_history": "temperature history is available",
    "scene_access": "access to the body by insects has been reconstructed",
    "body_movement_history": "body movement history has been evaluated",
    "adult_activity": "adult fly activity has been documented",
    "barrier_context": "covering/wrapping/barrier context has been evaluated",
    "collection_context": "collection/sampling context has been documented",
    "direct_scene_statement": "the statement is directly visible or explicitly provided",
}


@dataclass
class Claim:
    claim_id: str
    text: str
    initial_best_label: str
    final_best_label_if_all_needed_evidence_available: str
    required_tags_for_defensible_inference: Set[str]
    explanation_initial: str
    explanation_after_evidence: str
    overreach_warning: str


@dataclass
class EvidenceItem:
    item_id: str
    name: str
    cost: int
    tags: Set[str]
    text: str
    teaching_note: str


@dataclass
class Case:
    case_id: str
    title: str
    short_title: str
    initial_scene: str
    direct_observations: List[str]
    evidence_items: List[EvidenceItem]
    claims: List[Claim]


CASES: Dict[str, Case] = {
    "porch": Case(
        case_id="porch",
        title="Case 1: Porch Discovery — Larvae, Cloth, and a Dangerous Temptation",
        short_title="Larvae on remains",
        initial_scene=(
            "A deceased small mammal is found on a shaded porch. Several pale larvae are visible near the mouth opening. "
            "The body is partially covered by a cloth. No adult flies are visible in the photograph. The temperature at discovery is 26°C."
        ),
        direct_observations=[
            "Several pale larvae are visible near the mouth opening.",
            "The body is partially covered by a cloth.",
            "The remains are on a shaded porch.",
            "No adult flies are visible in the photograph.",
            "The temperature at discovery is 26°C.",
        ],
        evidence_items=[
            EvidenceItem(
                "species",
                "Request species identification",
                2,
                {"species_id"},
                "The larvae are identified as a blow fly species commonly used in forensic developmental estimates.",
                "Species identity matters because developmental rates differ among taxa.",
            ),
            EvidenceItem(
                "stage",
                "Request developmental staging",
                2,
                {"developmental_stage", "larval_measurement"},
                "The larvae are late second instars, and a subsample has been measured and preserved.",
                "Developmental stage and size move the evidence closer to a minimum colonization interval.",
            ),
            EvidenceItem(
                "temperature",
                "Request 72-hour temperature reconstruction",
                3,
                {"temperature_history"},
                "Nearby logger and weather station data suggest the porch microclimate averaged 24–28°C over the previous 72 hours.",
                "A single discovery temperature is not enough; accumulated thermal history is the relevant evidence.",
            ),
            EvidenceItem(
                "cloth",
                "Evaluate the cloth as an access barrier",
                2,
                {"barrier_context", "scene_access"},
                "The cloth covered the torso but did not fully cover the head or mouth opening. The mouth opening remained accessible.",
                "Partial covering can delay or redirect colonization, but the specific access path matters.",
            ),
            EvidenceItem(
                "adult",
                "Review scene video for adult fly activity",
                1,
                {"adult_activity"},
                "Brief adult fly activity is visible in video taken during scene processing, but not in the still photograph.",
                "Absence from a still image is not evidence of absence over time.",
            ),
            EvidenceItem(
                "movement",
                "Ask whether the body was moved",
                2,
                {"body_movement_history"},
                "The reporting party states the carcass was noticed in the same location the previous evening, but there is no earlier documentation.",
                "Scene history can constrain interpretation, but witness memory is not equivalent to continuous evidence.",
            ),
        ],
        claims=[
            Claim(
                "c1",
                "Larvae are visible near the mouth opening.",
                "Observation",
                "Observation",
                {"direct_scene_statement"},
                "This is directly provided by the initial scene description.",
                "This remains an observation even after additional evidence is collected.",
                "Do not inflate direct observations into time-of-death claims.",
            ),
            Claim(
                "c2",
                "The remains were accessible to ovipositing flies at some point before discovery.",
                "Inference",
                "Inference",
                {"visible_larvae", "scene_access"},
                "The visible larvae support some prior insect access, but the exact timing and route of access are not yet known.",
                "If access evidence confirms the mouth was not fully blocked, this becomes a stronger inference.",
                "This does not yet establish when death occurred.",
            ),
            Claim(
                "c3",
                "The larvae can support a minimum time-since-colonization estimate.",
                "Unsupported claim",
                "Inference",
                {"species_id", "developmental_stage", "larval_measurement", "temperature_history"},
                "Initially this is not defensible because visible larvae alone do not establish species, age, or accumulated temperature.",
                "With species ID, developmental stage, measurements, and temperature history, this becomes a defensible conditional inference.",
                "Minimum time since colonization is not automatically the same as time since death.",
            ),
            Claim(
                "c4",
                "The animal died exactly 48 hours before discovery.",
                "Unsupported claim",
                "Unsupported claim",
                {"species_id", "developmental_stage", "larval_measurement", "temperature_history", "scene_access", "body_movement_history"},
                "Exact PMI is not supported by the initial evidence.",
                "Even with more evidence, an exact time of death is usually overconfident; developmental evidence supports a range and often a minimum estimate.",
                "Exactness is the red flag: forensic evidence usually narrows uncertainty rather than eliminating it.",
            ),
            Claim(
                "c5",
                "The cloth may have delayed or altered insect colonization.",
                "Inference",
                "Inference",
                {"barrier_context", "scene_access"},
                "The cloth is visible, so barrier effects are plausible, but the direction and strength of the effect are not proven initially.",
                "If the cloth is shown to cover some access points but not others, this becomes a more disciplined inference.",
                "Do not turn 'may have delayed' into 'definitely delayed by X hours.'",
            ),
            Claim(
                "c6",
                "Because no adult flies are visible in the photograph, no adult flies visited the remains.",
                "Unsupported claim",
                "Unsupported claim",
                {"adult_activity"},
                "A still photograph only documents what is visible at that moment.",
                "Even if scene video shows adult flies later, the original claim remains overextended because it treats non-visibility as absence across time.",
                "Absence in one image is not absence from the scene history.",
            ),
        ],
    ),
    "window": Case(
        case_id="window",
        title="Case 2: Indoor Puparia — Migration, Development, and Location Overreach",
        short_title="Puparia near a window",
        initial_scene=(
            "A decomposing body is discovered indoors. Several empty puparia are found along the baseboard near a closed window. "
            "No larvae are visible on the body in the photograph. The room is described as warm, but no continuous temperature record is available."
        ),
        direct_observations=[
            "Several empty puparia are found near the baseboard by a window.",
            "The window is closed at discovery.",
            "No larvae are visible on the body in the photograph.",
            "The room is described as warm.",
            "No continuous temperature record is available.",
        ],
        evidence_items=[
            EvidenceItem(
                "species",
                "Identify puparia to species",
                2,
                {"species_id"},
                "The puparia are consistent with a calliphorid fly species with known developmental data.",
                "Species identity determines which developmental data can be used.",
            ),
            EvidenceItem(
                "stage",
                "Determine status of puparia",
                2,
                {"developmental_stage"},
                "The puparia are empty, indicating adult emergence occurred before discovery.",
                "Empty puparia support completion of immature development, but not necessarily location of death.",
            ),
            EvidenceItem(
                "temperature",
                "Request indoor temperature reconstruction",
                3,
                {"temperature_history"},
                "HVAC records suggest the room fluctuated between 23–25°C during the likely period of insect development.",
                "Indoor scenes can still have variable thermal histories that affect development.",
            ),
            EvidenceItem(
                "window_history",
                "Check window access history",
                2,
                {"scene_access"},
                "The window was closed at discovery, but neighbors reported it was open several days earlier.",
                "Discovery state should not be mistaken for the entire scene history.",
            ),
            EvidenceItem(
                "movement",
                "Evaluate possible body movement",
                3,
                {"body_movement_history"},
                "There is no direct evidence proving the body was always in this room. Transfer from another room remains possible.",
                "Insect evidence can support timelines, but location claims require independent scene context.",
            ),
            EvidenceItem(
                "collection",
                "Review collection notes",
                1,
                {"collection_context"},
                "Only puparia near the window were collected; the area under furniture was not searched.",
                "Sampling gaps should weaken broad claims about absence or distribution.",
            ),
        ],
        claims=[
            Claim(
                "w1",
                "Empty puparia are present near the window.",
                "Observation",
                "Observation",
                {"direct_scene_statement"},
                "This is directly stated in the scene description.",
                "This remains a direct observation.",
                "Do not confuse where puparia are found with where feeding occurred.",
            ),
            Claim(
                "w2",
                "Some larvae likely migrated away from the body before pupation.",
                "Inference",
                "Inference",
                {"visible_puparia", "developmental_stage", "collection_context"},
                "Puparia away from the body make migration plausible, but collection context and developmental status matter.",
                "Empty puparia plus location away from the body can support migration as a reasonable inference.",
                "Likely migration is not the same as a complete reconstruction of movement path.",
            ),
            Claim(
                "w3",
                "The body definitely decomposed in this exact room from the time of death onward.",
                "Unsupported claim",
                "Unsupported claim",
                {"body_movement_history", "scene_access", "temperature_history"},
                "Initial insect location does not prove the full location history of the body.",
                "Even after additional evidence, 'definitely from the time of death onward' remains too strong unless independent scene evidence establishes it.",
                "Words like definitely, always, and exact are often overreach markers.",
            ),
            Claim(
                "w4",
                "The empty puparia suggest that at least some insects completed immature development before discovery.",
                "Inference",
                "Inference",
                {"visible_puparia", "developmental_stage"},
                "Empty puparia are consistent with completed development, but species and status confirmation improve the claim.",
                "Once confirmed as empty puparia, the inference is strong.",
                "This supports development before discovery, not necessarily time since death.",
            ),
            Claim(
                "w5",
                "The closed window proves insects could not have entered from outside.",
                "Unsupported claim",
                "Unsupported claim",
                {"scene_access"},
                "The window was closed at discovery, but that does not establish its prior state.",
                "If evidence shows it was open earlier, the original claim collapses; if not, it may still be too strong because other access routes exist.",
                "Discovery state is not automatically historical state.",
            ),
            Claim(
                "w6",
                "A developmental timeline would require species identity and temperature history.",
                "Inference",
                "Inference",
                {"species_id", "temperature_history"},
                "This is a methodological inference based on how insect development is interpreted.",
                "Species identity and temperature history directly strengthen developmental timing.",
                "This is about what evidence is needed, not what happened in the case.",
            ),
        ],
    ),
    "wrapped": Case(
        case_id="wrapped",
        title="Case 3: Wrapped Remains — When Absence Is Not Absence",
        short_title="No visible insects",
        initial_scene=(
            "A decomposing body is found wrapped in a tarp in a basement. No larvae are visible in the photograph. "
            "There is strong odor and visible fluid staining beneath the body. The basement door was reportedly closed, "
            "but the reliability of that history is unknown."
        ),
        direct_observations=[
            "The body is wrapped in a tarp.",
            "The body is in a basement.",
            "No larvae are visible in the photograph.",
            "Strong odor is reported.",
            "Fluid staining is visible beneath the body.",
        ],
        evidence_items=[
            EvidenceItem(
                "tarp",
                "Inspect inside and under the tarp",
                2,
                {"barrier_context", "collection_context"},
                "Small larvae and eggs are found under a fold of the tarp near the head, but they were not visible in the original photograph.",
                "The photograph did not sample the full evidence space.",
            ),
            EvidenceItem(
                "species",
                "Identify recovered insects",
                2,
                {"species_id"},
                "Recovered larvae are identified as a fly taxon associated with decomposing tissue.",
                "Species identity helps distinguish forensic relevance from incidental arthropods.",
            ),
            EvidenceItem(
                "stage",
                "Stage and measure recovered larvae",
                2,
                {"developmental_stage", "larval_measurement"},
                "The recovered larvae are early instars with limited size range.",
                "Early instars support recent colonization, but the estimate still depends on temperature and access.",
            ),
            EvidenceItem(
                "temperature",
                "Request basement temperature history",
                3,
                {"temperature_history"},
                "The basement remained cool, approximately 18–20°C, during the relevant period.",
                "Cool temperatures can slow development relative to warmer conditions.",
            ),
            EvidenceItem(
                "access",
                "Evaluate insect access routes",
                2,
                {"scene_access"},
                "The basement has a floor drain and a poorly sealed exterior door threshold.",
                "Indoor or wrapped remains may still be accessible to insects through small entry routes.",
            ),
            EvidenceItem(
                "movement",
                "Evaluate movement history",
                3,
                {"body_movement_history"},
                "There is no direct evidence establishing whether the body was wrapped before or after death.",
                "Timing of wrapping is central to interpreting insect delay.",
            ),
        ],
        claims=[
            Claim(
                "r1",
                "No larvae are visible in the photograph.",
                "Observation",
                "Observation",
                {"direct_scene_statement"},
                "This is directly stated in the initial scene.",
                "This remains true even if hidden larvae are later found.",
                "The claim is about the photograph, not the entire body.",
            ),
            Claim(
                "r2",
                "No insects ever colonized the body.",
                "Unsupported claim",
                "Unsupported claim",
                {"collection_context", "barrier_context"},
                "A photograph showing no visible larvae cannot establish absence over the whole body or whole time period.",
                "If hidden larvae or eggs are found, the claim is directly contradicted.",
                "Absence claims require strong sampling evidence.",
            ),
            Claim(
                "r3",
                "The tarp may have delayed or limited insect colonization.",
                "Inference",
                "Inference",
                {"barrier_context", "scene_access"},
                "The tarp is a plausible access barrier, but the timing and completeness of that barrier are unknown initially.",
                "Inspection of tarp coverage and access routes can make this a stronger inference.",
                "May have delayed is appropriately cautious; definitely prevented is not.",
            ),
            Claim(
                "r4",
                "The body is decomposing.",
                "Inference",
                "Inference",
                {"direct_scene_statement"},
                "Odor and fluid staining strongly support decomposition, though they are not a full decomposition-stage analysis.",
                "Additional evidence may refine stage, but the basic inference is already strong.",
                "Do not use decomposition alone to infer a precise PMI.",
            ),
            Claim(
                "r5",
                "The basement prevented all insect activity.",
                "Unsupported claim",
                "Unsupported claim",
                {"scene_access", "collection_context"},
                "Basement location may reduce access, but it does not prove complete exclusion.",
                "If access routes or hidden larvae are found, this claim becomes even less defensible.",
                "Indoor scenes are not insect-proof by default.",
            ),
            Claim(
                "r6",
                "A strong insect-based PMI estimate requires inspection beyond the initial photograph.",
                "Inference",
                "Inference",
                {"collection_context", "species_id", "developmental_stage", "temperature_history"},
                "The initial photograph does not provide enough biological or sampling evidence.",
                "After targeted collection and temperature reconstruction, a more disciplined estimate may become possible.",
                "This is a claim about evidentiary sufficiency, not a claim about time of death.",
            ),
        ],
    ),
}

# -----------------------------
# Helper functions
# -----------------------------

def initialize_state() -> None:
    if "submitted_phase1" not in st.session_state:
        st.session_state.submitted_phase1 = {}
    if "submitted_phase2" not in st.session_state:
        st.session_state.submitted_phase2 = {}
    if "evidence_selected" not in st.session_state:
        st.session_state.evidence_selected = {}
    if "written_answers" not in st.session_state:
        st.session_state.written_answers = {}


def available_tags(case: Case, selected_ids: List[str]) -> Set[str]:
    tags = {"direct_scene_statement"}

    # Initial scene-level evidence tags.
    if case.case_id == "porch":
        tags.add("visible_larvae")
    if case.case_id == "window":
        tags.add("visible_puparia")
    if case.case_id == "wrapped":
        # Important: no visible larvae tag at start.
        pass

    for item in case.evidence_items:
        if item.item_id in selected_ids:
            tags |= item.tags
            if case.case_id == "wrapped" and item.item_id == "tarp":
                tags.add("visible_larvae")
    return tags


def evidence_cost(case: Case, selected_ids: List[str]) -> int:
    cost_lookup = {item.item_id: item.cost for item in case.evidence_items}
    return sum(cost_lookup[item_id] for item_id in selected_ids)


def claim_support_fraction(claim: Claim, tags: Set[str]) -> float:
    needed = claim.required_tags_for_defensible_inference
    if not needed:
        return 1.0
    return len(needed & tags) / len(needed)


def recommended_label_after_evidence(claim: Claim, tags: Set[str]) -> str:
    if claim.initial_best_label == "Observation":
        return "Observation"

    # Some claims remain overreach no matter what because their wording is absolute/exact.
    if claim.final_best_label_if_all_needed_evidence_available == "Unsupported claim":
        return "Unsupported claim"

    support = claim_support_fraction(claim, tags)
    if support >= 0.75:
        return claim.final_best_label_if_all_needed_evidence_available
    return claim.initial_best_label


def calibration_points(correct: bool, confidence: int) -> float:
    # Rewards confidence only when correct; punishes overconfidence when wrong.
    if correct:
        return {1: 0.4, 2: 0.6, 3: 0.8, 4: 1.0, 5: 1.0}[confidence]
    return {1: 0.5, 2: 0.3, 3: 0.0, 4: -0.5, 5: -1.0}[confidence]


def score_phase(case: Case, phase: str, tags: Set[str]) -> pd.DataFrame:
    rows = []
    for claim in case.claims:
        label_key = f"{case.case_id}_{phase}_{claim.claim_id}_label"
        conf_key = f"{case.case_id}_{phase}_{claim.claim_id}_conf"
        student_label = st.session_state.get(label_key, LABELS[0])
        confidence = st.session_state.get(conf_key, 3)
        if phase == "phase1":
            recommended = claim.initial_best_label
            explanation = claim.explanation_initial
        else:
            recommended = recommended_label_after_evidence(claim, tags)
            explanation = claim.explanation_after_evidence
        correct = student_label == recommended
        rows.append(
            {
                "Claim": claim.text,
                "Your classification": student_label,
                "Recommended classification": recommended,
                "Aligned?": "Yes" if correct else "No",
                "Confidence": confidence,
                "Calibration points": calibration_points(correct, confidence),
                "Support fraction": round(claim_support_fraction(claim, tags), 2),
                "Explanation": explanation,
                "Overreach warning": claim.overreach_warning,
            }
        )
    return pd.DataFrame(rows)


def phase_score(df: pd.DataFrame) -> Tuple[float, float]:
    accuracy = float((df["Aligned?"] == "Yes").mean()) if len(df) else 0.0
    calibration = float(df["Calibration points"].sum()) if len(df) else 0.0
    return accuracy, calibration


def selected_evidence_items(case: Case, selected_ids: List[str]) -> List[EvidenceItem]:
    return [item for item in case.evidence_items if item.item_id in selected_ids]


def missing_evidence_for_claim(claim: Claim, tags: Set[str]) -> List[str]:
    missing = list(claim.required_tags_for_defensible_inference - tags)
    return [EVIDENCE_TAG_DESCRIPTIONS.get(tag, tag) for tag in missing]


def format_support_bar(fraction: float) -> str:
    pct = int(fraction * 100)
    return f"{pct}%"

# -----------------------------
# App state and sidebar
# -----------------------------

initialize_state()

st.sidebar.title("🪰 W1 Evidence Ladder")
st.sidebar.markdown(
    "Students first reason from incomplete evidence, then choose which evidence to request, then revise their claims."
)

selected_case_id = st.sidebar.radio(
    "Choose case",
    options=list(CASES.keys()),
    format_func=lambda cid: CASES[cid].short_title,
)
case = CASES[selected_case_id]

st.sidebar.divider()
EVIDENCE_BUDGET = st.sidebar.slider("Evidence budget", min_value=3, max_value=10, value=6)
st.sidebar.markdown("Smaller budgets force students to prioritize which evidence matters most.")

st.sidebar.divider()
show_key = st.sidebar.checkbox("Instructor mode: show full key", value=False)

# -----------------------------
# Header
# -----------------------------

st.title("How Far Can the Evidence Take You?")
st.caption("W1 — Observation vs. Inference as a forensic reasoning problem")

st.markdown(
    "This app is not asking students to memorize labels. It asks them to make claims under uncertainty, "
    "decide what evidence would actually improve those claims, and then revise their confidence."
)

with st.expander("What counts as observation, inference, or unsupported claim?", expanded=False):
    st.markdown(
        "**Observation**: directly visible, measured, photographed, recorded, or explicitly stated.\n\n"
        "**Inference**: a defensible interpretation that follows from evidence, usually with stated assumptions.\n\n"
        "**Unsupported claim**: a claim that may be possible, but the current evidence does not justify saying it.\n\n"
        "The dangerous category is not 'wrong.' It is often 'not yet defensible.'"
    )

# -----------------------------
# Phase 1: Initial evidence only
# -----------------------------

st.header(case.title)

left, right = st.columns([1.0, 1.25], gap="large")

with left:
    st.subheader("Initial scene evidence")
    st.info(case.initial_scene)

    st.markdown("### Direct observations available at the start")
    for obs in case.direct_observations:
        st.markdown(f"- {obs}")

    st.warning(
        "At this stage, students do **not** know species ID, developmental stage, accumulated temperature, "
        "full access history, or whether the scene history is complete."
    )

with right:
    st.subheader("Phase 1 — Classify claims from incomplete evidence")
    st.markdown(
        "Classify each claim using only the initial scene. Then rate confidence. "
        "High confidence is good only when the claim is actually defensible."
    )

    for claim in case.claims:
        st.markdown(f"**{claim.text}**")
        c1, c2 = st.columns([0.68, 0.32])
        with c1:
            st.selectbox(
                "Initial classification",
                LABELS,
                key=f"{case.case_id}_phase1_{claim.claim_id}_label",
                label_visibility="collapsed",
            )
        with c2:
            st.select_slider(
                "Confidence",
                options=list(CONFIDENCE_LABELS.keys()),
                format_func=lambda x: CONFIDENCE_LABELS[x],
                value=3,
                key=f"{case.case_id}_phase1_{claim.claim_id}_conf",
                label_visibility="collapsed",
            )
        st.divider()

if st.button("Lock Phase 1 classifications", type="primary"):
    st.session_state.submitted_phase1[case.case_id] = True

if st.session_state.submitted_phase1.get(case.case_id, False):
    initial_tags = available_tags(case, [])
    phase1_df = score_phase(case, "phase1", initial_tags)
    acc1, cal1 = phase_score(phase1_df)

    st.subheader("Phase 1 feedback")
    m1, m2 = st.columns(2)
    m1.metric("Initial alignment", f"{acc1 * 100:.0f}%")
    m2.metric("Calibration score", f"{cal1:.1f}")

    st.dataframe(
        phase1_df[["Claim", "Your classification", "Recommended classification", "Aligned?", "Confidence", "Explanation", "Overreach warning"]],
        hide_index=True,
        use_container_width=True,
    )

# -----------------------------
# Phase 2: Evidence budget
# -----------------------------

st.header("Phase 2 — Spend your evidence budget")
st.markdown(
    "Now decide what additional evidence you would request. You cannot ask for everything. "
    "The point is to identify which missing evidence actually changes the defensibility of a claim."
)

selected_ids = []
current_cost = 0

e1, e2 = st.columns([1.1, 1.0], gap="large")

with e1:
    st.subheader("Evidence menu")
    for item in case.evidence_items:
        disabled = current_cost + item.cost > EVIDENCE_BUDGET
        checked = st.checkbox(
            f"{item.name} — cost {item.cost}",
            key=f"{case.case_id}_evidence_{item.item_id}",
            help=item.teaching_note,
        )
        if checked:
            selected_ids.append(item.item_id)
            current_cost += item.cost

    total_cost = evidence_cost(case, selected_ids)
    if total_cost > EVIDENCE_BUDGET:
        st.error(f"Evidence budget exceeded: {total_cost}/{EVIDENCE_BUDGET}. Uncheck one or more items.")
    else:
        st.success(f"Evidence selected: {total_cost}/{EVIDENCE_BUDGET} budget used.")

with e2:
    st.subheader("Evidence revealed")
    if total_cost <= EVIDENCE_BUDGET and selected_ids:
        for item in selected_evidence_items(case, selected_ids):
            st.markdown(f"**{item.name}**")
            st.markdown(item.text)
            st.caption(item.teaching_note)
            st.divider()
    elif not selected_ids:
        st.info("Select evidence from the menu to reveal it.")
    else:
        st.warning("Reduce your evidence selection to stay within budget.")

# Store selected evidence if valid.
if total_cost <= EVIDENCE_BUDGET:
    st.session_state.evidence_selected[case.case_id] = selected_ids

tags_after = available_tags(case, st.session_state.evidence_selected.get(case.case_id, []))

# -----------------------------
# Evidence impact dashboard
# -----------------------------

st.subheader("What did your evidence actually support?")
impact_rows = []
for claim in case.claims:
    support = claim_support_fraction(claim, tags_after)
    missing = missing_evidence_for_claim(claim, tags_after)
    impact_rows.append(
        {
            "Claim": claim.text,
            "Evidence support": format_support_bar(support),
            "Current recommended status": recommended_label_after_evidence(claim, tags_after),
            "Still missing": "; ".join(missing) if missing else "No major required evidence missing",
        }
    )
st.dataframe(pd.DataFrame(impact_rows), hide_index=True, use_container_width=True)

# -----------------------------
# Phase 3: Revision
# -----------------------------

st.header("Phase 3 — Revise after evidence")
st.markdown(
    "Now classify the claims again. A good revision is not necessarily changing every answer. "
    "A good revision is changing only the claims whose evidentiary support actually changed."
)

for claim in case.claims:
    st.markdown(f"**{claim.text}**")
    c1, c2 = st.columns([0.68, 0.32])
    with c1:
        st.selectbox(
            "Revised classification",
            LABELS,
            key=f"{case.case_id}_phase2_{claim.claim_id}_label",
            label_visibility="collapsed",
        )
    with c2:
        st.select_slider(
            "Revised confidence",
            options=list(CONFIDENCE_LABELS.keys()),
            format_func=lambda x: CONFIDENCE_LABELS[x],
            value=3,
            key=f"{case.case_id}_phase2_{claim.claim_id}_conf",
            label_visibility="collapsed",
        )
    st.divider()

st.subheader("Written reasoning")
st.markdown("These responses are the pieces I would grade most heavily.")

strongest = st.text_area(
    "1. Which claim is now the strongest defensible inference, and why?",
    key=f"{case.case_id}_strongest",
    height=110,
    placeholder="Name the claim. Explain what evidence supports it and what assumptions remain.",
)

overreach = st.text_area(
    "2. Which claim still goes beyond the evidence, even after your evidence requests?",
    key=f"{case.case_id}_overreach",
    height=110,
    placeholder="Name the claim. Explain why the wording or missing evidence makes it overextended.",
)

missing = st.text_area(
    "3. What specific additional evidence would be needed to make one currently unsupported claim defensible?",
    key=f"{case.case_id}_missing",
    height=110,
    placeholder="Be specific: species, larval age, temperature history, access history, body movement history, collection context, etc.",
)

if st.button("Submit final reasoning", type="primary"):
    st.session_state.submitted_phase2[case.case_id] = True
    st.session_state.written_answers[case.case_id] = {
        "strongest": strongest,
        "overreach": overreach,
        "missing": missing,
    }

# -----------------------------
# Final feedback
# -----------------------------

if st.session_state.submitted_phase2.get(case.case_id, False):
    st.header("Final feedback")
    phase2_df = score_phase(case, "phase2", tags_after)
    acc2, cal2 = phase_score(phase2_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Revised alignment", f"{acc2 * 100:.0f}%")
    c2.metric("Revised calibration", f"{cal2:.1f}")
    c3.metric("Evidence budget used", f"{evidence_cost(case, st.session_state.evidence_selected.get(case.case_id, []))}/{EVIDENCE_BUDGET}")

    st.markdown("### Claim audit")
    st.dataframe(
        phase2_df[["Claim", "Your classification", "Recommended classification", "Aligned?", "Confidence", "Support fraction", "Explanation", "Overreach warning"]],
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Evidence discipline score guide")
    st.markdown(
        "A strong answer does three things: "
        "1) separates what is directly visible from what is interpreted, "
        "2) recognizes when new evidence changes claim status, and "
        "3) refuses to convert insect evidence into a precise death narrative unless the missing assumptions are addressed."
    )

    st.markdown("### Suggested 10-point grading")
    grading_df = pd.DataFrame(
        [
            {"Component": "Initial classification and calibration", "Points": 2, "What earns credit": "Reasonable initial labels; low confidence when evidence is incomplete."},
            {"Component": "Evidence selection", "Points": 2, "What earns credit": "Requests evidence that actually changes defensibility, not just interesting extras."},
            {"Component": "Revised classification", "Points": 2, "What earns credit": "Changes claims only when new evidence supports the change."},
            {"Component": "Strongest inference explanation", "Points": 2, "What earns credit": "Names a defensible inference and explains the evidence and remaining assumptions."},
            {"Component": "Overreach and missing evidence", "Points": 2, "What earns credit": "Identifies an unsupported claim and states the specific evidence needed to support it."},
        ]
    )
    st.dataframe(grading_df, hide_index=True, use_container_width=True)

    export_rows = []
    export_rows.append({"Section": "Case", "Item": case.title, "Response": ""})
    export_rows.append({"Section": "Evidence selected", "Item": ", ".join(st.session_state.evidence_selected.get(case.case_id, [])), "Response": ""})
    for _, row in phase2_df.iterrows():
        export_rows.append({"Section": "Claim audit", "Item": row["Claim"], "Response": f"Student: {row['Your classification']} | Recommended: {row['Recommended classification']} | Confidence: {row['Confidence']}"})
    answers = st.session_state.written_answers.get(case.case_id, {})
    export_rows.append({"Section": "Written", "Item": "Strongest defensible inference", "Response": answers.get("strongest", "")})
    export_rows.append({"Section": "Written", "Item": "Overextended claim", "Response": answers.get("overreach", "")})
    export_rows.append({"Section": "Written", "Item": "Additional evidence needed", "Response": answers.get("missing", "")})
    export_df = pd.DataFrame(export_rows)
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download submission CSV",
        data=csv,
        file_name=f"{case.case_id}_evidence_ladder_submission.csv",
        mime="text/csv",
    )

# -----------------------------
# Instructor key
# -----------------------------

if show_key:
    st.header("Instructor key")

    st.subheader("Evidence tag descriptions")
    tag_df = pd.DataFrame(
        [{"Tag": tag, "Meaning": desc} for tag, desc in EVIDENCE_TAG_DESCRIPTIONS.items()]
    )
    st.dataframe(tag_df, hide_index=True, use_container_width=True)

    st.subheader("Claim logic")
    key_rows = []
    for claim in case.claims:
        key_rows.append(
            {
                "Claim": claim.text,
                "Initial best label": claim.initial_best_label,
                "Final if supported": claim.final_best_label_if_all_needed_evidence_available,
                "Required evidence tags": ", ".join(sorted(claim.required_tags_for_defensible_inference)),
                "Initial explanation": claim.explanation_initial,
                "After-evidence explanation": claim.explanation_after_evidence,
                "Overreach warning": claim.overreach_warning,
            }
        )
    st.dataframe(pd.DataFrame(key_rows), hide_index=True, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------

st.divider()
st.markdown(
    "**Instructor note:** This app should produce argument, not clicking. The best classroom use is to pause after Phase 1, "
    "ask students which claims they were confident about, then reveal how few of those claims actually had the evidence needed."
)
