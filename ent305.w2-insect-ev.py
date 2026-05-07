# W2 — Insect Presence Claim-Calibration App
# ENT 305: Forensic Entomology and the Ecology of Decay
#
# Purpose:
# Students practice distinguishing insect presence from forensic time evidence.
# The app asks: what does the observation show, what does it suggest, and where must inference stop?
#
# Run with:
# streamlit run w2_insect_presence_claim_calibration.py

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Insect Presence Claim Calibration",
    page_icon="🪰",
    layout="wide",
)

CLAIM_LEVELS = [
    "Direct observation",
    "Cautious ecological inference",
    "Possible colonization evidence",
    "Possible developmental/time evidence",
    "Unsupported overclaim",
]

MISSING_EVIDENCE_OPTIONS = [
    "Species identification",
    "Developmental stage or age class",
    "Larval measurements",
    "Eggs or larvae, not just adults",
    "Temperature history",
    "Access context: exposed, wrapped, buried, indoor, screened, sealed",
    "Scene history: whether the body was moved or access changed",
    "Collection notes and specimen preservation",
    "Comparison to validated developmental data",
    "No additional evidence needed for a PMI-related claim",
]

CASES = [
    {
        "id": "adult_flies",
        "title": "Adult flies on exposed remains",
        "scene": "A body is found outdoors in an exposed location. Several adult flies are seen landing on the face and hands. No eggs or larvae are visible in the photograph.",
        "observation": "Adult flies are present on the remains.",
        "strongest": "Cautious ecological inference",
        "inference": "The remains are attractive and accessible to adult flies at the time of observation.",
        "overclaim": "The flies prove that colonization has already occurred or that a PMI can be estimated.",
        "needed": ["Eggs or larvae, not just adults", "Species identification", "Developmental stage or age class", "Temperature history", "Access context: exposed, wrapped, buried, indoor, screened, sealed"],
        "feedback": "Adult fly presence can support attraction and access. By itself, it is not developmental evidence and does not establish colonization timing.",
    },
    {
        "id": "eggs_orifice",
        "title": "Eggs near an orifice",
        "scene": "A body is found outdoors. A small cluster of pale eggs is visible near the nostril. No larvae are visible yet. The scene is exposed and adult flies are nearby.",
        "observation": "Eggs are visible near the nostril.",
        "strongest": "Possible colonization evidence",
        "inference": "The remains may have been colonized by ovipositing insects before discovery.",
        "overclaim": "The eggs establish the time of death or a precise PMI.",
        "needed": ["Species identification", "Developmental stage or age class", "Temperature history", "Access context: exposed, wrapped, buried, indoor, screened, sealed", "Comparison to validated developmental data"],
        "feedback": "Eggs are stronger than adult presence because they may indicate colonization. But eggs alone still do not give a precise PMI without species and temperature context.",
    },
    {
        "id": "larvae_wound",
        "title": "Larvae feeding in a wound",
        "scene": "Larvae are visibly feeding in an open wound on the body. The body is outdoors and accessible. The photograph shows active larval masses, but no species identification has been performed.",
        "observation": "Larvae are feeding in an open wound.",
        "strongest": "Possible developmental/time evidence",
        "inference": "The larvae may provide developmental evidence if they are properly identified, staged, measured, and interpreted with temperature history.",
        "overclaim": "The wound was the cause of death, or the larvae prove the exact time of death.",
        "needed": ["Species identification", "Developmental stage or age class", "Larval measurements", "Temperature history", "Collection notes and specimen preservation", "Comparison to validated developmental data"],
        "feedback": "Larvae are the first scenario in this app that may support developmental/time evidence, but only after identification, staging, measurement, and thermal context.",
    },
    {
        "id": "beetles_dry",
        "title": "Beetles on dry tissue",
        "scene": "Several beetles are present on dry skin and connective tissue. The body is outdoors and appears partially skeletonized. No larvae are described in the scene notes.",
        "observation": "Beetles are present on dry tissue.",
        "strongest": "Cautious ecological inference",
        "inference": "The remains include dry tissue that is being used by beetles, which may be consistent with later decomposition conditions.",
        "overclaim": "The beetles alone establish a precise PMI.",
        "needed": ["Species identification", "Scene history: whether the body was moved or access changed", "Access context: exposed, wrapped, buried, indoor, screened, sealed", "Temperature history"],
        "feedback": "Beetle presence may suggest ecological conditions around the remains, especially dryness, but it is not automatically precise developmental time evidence.",
    },
    {
        "id": "no_insects_indoors",
        "title": "No visible insects indoors",
        "scene": "A decomposing body is found indoors in a closed room. No insects are visible in the photograph. The window is closed at discovery.",
        "observation": "No insects are visible in the photograph.",
        "strongest": "Direct observation",
        "inference": "The photograph does not show visible insects at the time and angle documented.",
        "overclaim": "No insects ever accessed the body, or insect evidence is impossible in this case.",
        "needed": ["Access context: exposed, wrapped, buried, indoor, screened, sealed", "Scene history: whether the body was moved or access changed", "Collection notes and specimen preservation"],
        "feedback": "No visible insects in a photograph is a narrow observation. It does not prove absence across the whole scene or the whole postmortem interval.",
    },
    {
        "id": "pupae_away",
        "title": "Pupae found away from the body",
        "scene": "Several pupae and empty puparia are found along the edge of the room, several meters away from the body. No larvae are visible on the body in the photograph.",
        "observation": "Pupae or empty puparia are present away from the body.",
        "strongest": "Possible developmental/time evidence",
        "inference": "The pupae may indicate that insects developed on or near the remains and later moved away before pupation or adult emergence.",
        "overclaim": "The body has been in this room since death, or the puparia alone prove an exact PMI.",
        "needed": ["Species identification", "Developmental stage or age class", "Temperature history", "Scene history: whether the body was moved or access changed", "Collection notes and specimen preservation", "Comparison to validated developmental data"],
        "feedback": "Pupae away from the body can be important, but they require careful interpretation. They may be developmental evidence, but they do not by themselves prove the whole location history or exact PMI.",
    },
    {
        "id": "wrapped_larvae",
        "title": "Larvae on wrapped remains",
        "scene": "A body is found partly wrapped in a blanket. Larvae are visible only near an exposed part of the face. Most of the body remains covered.",
        "observation": "Larvae are visible near the exposed part of the face.",
        "strongest": "Possible developmental/time evidence",
        "inference": "The larvae may provide developmental evidence for colonization of the accessible region, but the wrapping may complicate interpretation of the whole body.",
        "overclaim": "The larval age directly equals time since death for the whole body.",
        "needed": ["Species identification", "Developmental stage or age class", "Larval measurements", "Temperature history", "Access context: exposed, wrapped, buried, indoor, screened, sealed", "Scene history: whether the body was moved or access changed"],
        "feedback": "Larvae can support time-related inference, but wrapping creates a colonization-access problem. Developmental age is not automatically time since death.",
    },
]

# -----------------------------
# Header
# -----------------------------

st.title("Insect Presence Claim Calibration")
st.caption("What does insect presence show, what does it suggest, and where should inference stop?")

st.markdown(
    "For each scene, choose the strongest type of claim the observation can support. "
    "Then identify the direct observation, the strongest defensible inference, the overclaim to avoid, "
    "and the evidence needed before making a PMI-related claim."
)

with st.expander("Claim ladder", expanded=True):
    ladder_df = pd.DataFrame(
        [
            {
                "Claim level": "Direct observation",
                "Meaning": "What is directly visible or stated.",
                "Example": "Adult flies are present on the remains.",
            },
            {
                "Claim level": "Cautious ecological inference",
                "Meaning": "A modest interpretation about access, attraction, habitat, or condition.",
                "Example": "The remains are attractive and accessible to flies.",
            },
            {
                "Claim level": "Possible colonization evidence",
                "Meaning": "Evidence that insects may have begun using the body as a reproductive or feeding resource.",
                "Example": "Eggs near an orifice may indicate oviposition.",
            },
            {
                "Claim level": "Possible developmental/time evidence",
                "Meaning": "Evidence that could contribute to timing, but only with species, stage, temperature, and access context.",
                "Example": "Larval stage may help estimate minimum time since colonization.",
            },
            {
                "Claim level": "Unsupported overclaim",
                "Meaning": "A claim that exceeds what the evidence can support.",
                "Example": "Adult flies prove the person died two days ago.",
            },
        ]
    )
    st.dataframe(ladder_df, hide_index=True, use_container_width=True)

# -----------------------------
# Case interface
# -----------------------------

case_index = st.selectbox(
    "Choose a scene",
    range(len(CASES)),
    format_func=lambda i: CASES[i]["title"],
)
case = CASES[case_index]

st.header(case["title"])
st.info(case["scene"])

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Your claim calibration")

    chosen_level = st.radio(
        "What is the strongest claim level this observation can support?",
        CLAIM_LEVELS,
        key=f"{case['id']}_level",
    )

    direct_observation = st.text_area(
        "1. What is the direct observation?",
        height=80,
        key=f"{case['id']}_observation",
        placeholder="State only what is directly visible or stated.",
    )

    strongest_inference = st.text_area(
        "2. What is the strongest defensible inference?",
        height=90,
        key=f"{case['id']}_inference",
        placeholder="Make the strongest claim that still stays tied to the evidence.",
    )

with right:
    st.subheader("Where inference must stop")

    overclaim = st.text_area(
        "3. What would be overreach?",
        height=90,
        key=f"{case['id']}_overclaim",
        placeholder="Name a claim that would go beyond the evidence.",
    )

    needed = st.multiselect(
        "4. What additional evidence would be needed before making a PMI-related claim?",
        MISSING_EVIDENCE_OPTIONS,
        key=f"{case['id']}_needed",
    )

    short_why = st.text_area(
        "Explain your reasoning in one or two sentences.",
        height=100,
        key=f"{case['id']}_why",
        placeholder="Because...",
    )

# -----------------------------
# Feedback
# -----------------------------

if st.button("Check my work", type="primary"):
    level_ok = chosen_level == case["strongest"]

    selected_needed = set(needed)
    expected_needed = set(case["needed"])
    correct_needed = selected_needed & expected_needed
    missed_needed = expected_needed - selected_needed
    extra_needed = selected_needed - expected_needed

    needed_score = len(correct_needed) / len(expected_needed) if expected_needed else 1.0

    st.header("Feedback")

    c1, c2 = st.columns(2)
    c1.metric("Claim level", "Matched" if level_ok else "Needs revision")
    c2.metric("PMI evidence match", f"{needed_score * 100:.0f}%")

    if level_ok:
        st.success("Your selected claim level matches the recommended calibration.")
    else:
        st.warning(f"Recommended strongest claim level: **{case['strongest']}**")

    st.subheader("Reasoning ladder for this scene")
    ladder = pd.DataFrame(
        [
            {"Step": "Observation", "Recommended answer": case["observation"]},
            {"Step": "Strongest defensible inference", "Recommended answer": case["inference"]},
            {"Step": "Overclaim to avoid", "Recommended answer": case["overclaim"]},
            {"Step": "Why", "Recommended answer": case["feedback"]},
        ]
    )
    st.dataframe(ladder, hide_index=True, use_container_width=True)

    st.subheader("PMI-related evidence")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Useful evidence you selected**")
        if correct_needed:
            for item in sorted(correct_needed):
                st.markdown(f"- {item}")
        else:
            st.markdown("None selected.")
    with col_b:
        st.markdown("**Important evidence missed**")
        if missed_needed:
            for item in sorted(missed_needed):
                st.markdown(f"- {item}")
        else:
            st.markdown("No major expected items missed.")
    with col_c:
        st.markdown("**Less relevant selections**")
        if extra_needed:
            for item in sorted(extra_needed):
                st.markdown(f"- {item}")
        else:
            st.markdown("No extra items selected.")

    export = pd.DataFrame(
        [
            {"Prompt": "Scene", "Response": case["title"]},
            {"Prompt": "Selected claim level", "Response": chosen_level},
            {"Prompt": "Recommended claim level", "Response": case["strongest"]},
            {"Prompt": "Direct observation", "Response": direct_observation},
            {"Prompt": "Recommended observation", "Response": case["observation"]},
            {"Prompt": "Strongest inference", "Response": strongest_inference},
            {"Prompt": "Recommended inference", "Response": case["inference"]},
            {"Prompt": "Overclaim", "Response": overclaim},
            {"Prompt": "Recommended overclaim", "Response": case["overclaim"]},
            {"Prompt": "PMI evidence selected", "Response": "; ".join(needed)},
            {"Prompt": "Expected PMI evidence", "Response": "; ".join(case["needed"])},
            {"Prompt": "Explanation", "Response": short_why},
        ]
    )

    csv = export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download feedback as CSV",
        data=csv,
        file_name=f"{case['id']}_claim_calibration_feedback.csv",
        mime="text/csv",
    )
