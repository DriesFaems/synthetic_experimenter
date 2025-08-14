
import os
import json
import math
import random
import time
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional Groq import (only loaded when needed)
try:
    from groq import Groq  # type: ignore
except Exception:
    Groq = None


# -----------------------------
# Utilities
# -----------------------------

@dataclass
class AppConfig:
    default_model: str = "openai/gpt-oss-20b"  # Default to the requested model
    temperature: float = 0.2
    max_tokens: int = 1024
    seed: int = 42
    n_permutations: int = 5000
    n_bootstrap: int = 3000


RNG = np.random.default_rng(42)


def set_seed(seed: int):
    global RNG
    RNG = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_country_pool(geo_focus: str) -> List[str]:
    """Return a pool of nationalities consistent with the chosen geography.
    We keep it simple and synthetic by using country names as 'nationalities'."""
    germany = ["German"]
    europe = [
        "Albanian", "Austrian", "Belgian", "Bosnian", "Bulgarian", "Croatian", "Cypriot",
        "Czech", "Danish", "Dutch", "English", "Estonian", "Finnish", "French",
        "German", "Greek", "Hungarian", "Icelandic", "Irish", "Italian", "Latvian",
        "Lithuanian", "Luxembourgish", "Maltese", "Montenegrin", "Norwegian", "Polish",
        "Portuguese", "Romanian", "Scottish", "Serbian", "Slovak", "Slovenian", "Spanish",
        "Swedish", "Swiss", "Turkish", "Ukrainian", "Welsh"
    ]
    global_pool = europe + [
        "American", "Argentine", "Australian", "Bangladeshi", "Bolivian", "Brazilian",
        "Canadian", "Chilean", "Chinese", "Colombian", "Costa Rican", "Dominican",
        "Ecuadorian", "Egyptian", "Ethiopian", "Filipino", "Ghanaian", "Guatemalan",
        "Hong Kong", "Indian", "Indonesian", "Iranian", "Iraqi", "Israeli", "Japanese",
        "Jordanian", "Kenyan", "Korean", "Kuwaiti", "Lebanese", "Malaysian", "Mexican",
        "Moroccan", "Nigerian", "Pakistani", "Peruvian", "Qatari", "Russian", "Saudi",
        "Singaporean", "South African", "Sri Lankan", "Taiwanese", "Tanzanian", "Thai",
        "Tunisian", "Uruguayan", "Venezuelan", "Vietnamese"
    ]
    if geo_focus.lower() == "germany":
        return germany
    if geo_focus.lower() == "europe":
        return europe
    return global_pool


def sample_respondents(n: int, age_min: int, age_max: int, geo_focus: str) -> pd.DataFrame:
    nationalities = get_country_pool(geo_focus)
    genders = ["Female", "Male", "Non-binary"]
    ages = RNG.integers(low=age_min, high=age_max + 1, size=n)
    gender_choices = RNG.choice(genders, size=n, replace=True)
    nationality_choices = RNG.choice(nationalities, size=n, replace=True)

    df = pd.DataFrame({
        "respondent_id": np.arange(1, n + 1),
        "age": ages,
        "gender": gender_choices,
        "nationality": nationality_choices
    })
    return df


def persona_prompt(profile: str,
                   target_segment: str,
                   question: str,
                   scenario_a: str,
                   scenario_b: str,
                   response_mode: str,
                   scale_max: int) -> List[Dict[str, str]]:
    """Build the chat messages for LLM API calls."""
    if response_mode == "Likert":
        response_instructions = f"""
Return ONLY valid minified JSON on one line with the following schema:
{{
  "persona": "<3-5 sentence persona description>",
  "scenario_A": {{"answer_text":"<short answer>", "rating": <integer 1-{scale_max}>}},
  "scenario_B": {{"answer_text":"<short answer>", "rating": <integer 1-{scale_max}>}}
}}

Rules:
- The 'rating' MUST be an INTEGER between 1 and {scale_max}, inclusive.
- DO NOT include markdown fences or commentary.
- Keep answers concise (<= 40 words each).
"""
    else:
        response_instructions = """
Return ONLY valid minified JSON on one line with the following schema:
{
  "persona": "<3-5 sentence persona description>",
  "scenario_A": {"answer_text":"<short answer>", "choice": "Yes"|"No"},
  "scenario_B": {"answer_text":"<short answer>", "choice": "Yes"|"No"}
}

Rules:
- 'choice' MUST be exactly "Yes" or "No".
- DO NOT include markdown fences or commentary.
- Keep answers concise (<= 40 words each).
"""
    system_content = f"""You are role-playing as a survey respondent. Here is your profile: {profile}
You belong to the target customer segment: {target_segment}.
Task: Carefully read the user's evaluation question and two alternative value propositions (A and B). Answer the question for both scenarios.
Be true to your profile. Avoid generic corporate jargon.
{response_instructions}
"""
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Evaluation question: {question}\n\nScenario A:\n{scenario_a}\n\nScenario B:\n{scenario_b}"}
    ]
    return messages


def clean_json(s: str) -> str:
    # Strip code fences and whitespace
    s = s.strip()
    s = re.sub(r"```json\s*|\s*```", "", s, flags=re.IGNORECASE)
    s = s.strip()
    # Attempt to cut leading/trailing characters before/after JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1:
        s = s[start:end+1]
    return s


def call_groq(api_key: str,
              model_name: str,
              messages: List[Dict[str, str]],
              temperature: float,
              max_tokens: int) -> str:
    """Call Groq API (supports both Llama and OpenAI models via Groq)."""
    if Groq is None:
        raise RuntimeError("The 'groq' package is not available in this environment.")
    os.environ["GROQ_API_KEY"] = api_key
    client = Groq()
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        stream=False,
        stop=None,
    )
    return resp.choices[0].message.content  # type: ignore


def parse_persona_json(raw: str) -> Optional[dict]:
    try:
        cleaned = clean_json(raw)
        return json.loads(cleaned)
    except Exception:
        return None


def paired_permutation_test(a: np.ndarray, b: np.ndarray, n_perms: int = 5000, seed: Optional[int] = None) -> float:
    """Two-sided paired permutation test on mean difference (B - A)."""
    rng = np.random.default_rng(seed)
    d = b - a
    d_obs = float(np.mean(d))
    if d.size == 0:
        return float("nan")
    signs = rng.choice([-1, 1], size=(n_perms, d.size), replace=True)
    perms = np.mean(signs * d, axis=1)
    p = (np.sum(np.abs(perms) >= abs(d_obs)) + 1) / (n_perms + 1)
    return float(p)


def bootstrap_ci_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 3000, alpha: float = 0.05, seed: Optional[int] = None) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = a.size
    dists = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        dists.append(float(np.mean(b[idx] - a[idx])))
    lower = float(np.quantile(dists, alpha/2))
    upper = float(np.quantile(dists, 1 - alpha/2))
    return lower, upper


def cohen_dz(a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    sd = np.std(d, ddof=1) if d.size > 1 else 0.0
    if sd == 0:
        return 0.0
    return float(np.mean(d) / sd)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar p-value (two-sided) using binomial tail.
    b = A=No, B=Yes; c = A=Yes, B=No in paired 2x2 table."""
    from math import comb
    n = b + c
    if n == 0:
        return float("nan")
    # two-sided exact p-value
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = 2 * p
    return min(1.0, float(p))


def assign_age_group(age: int) -> str:
    bins = [(0, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 200)]
    labels = ["<=24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
    for (lo, hi), lab in zip(bins, labels):
        if lo <= age <= hi:
            return lab
    return "Other"


def plot_mean_bars(labels: List[str], means: List[float], title: str, ylabel: str):
    plt.figure()
    plt.bar(labels, means)
    plt.title(title)
    plt.ylabel(ylabel)
    st.pyplot(plt.gcf())
    plt.close()


def info_box(msg: str):
    st.info(msg)


# -----------------------------
# Streamlit App
# -----------------------------

def main():
    st.title("Synthetic A/B Experimenter")
    
    # Attribution
    st.caption("This application is developed by Dries Faems, for more applications go to the Gen AI Nerd Channel: https://www.youtube.com/@GenAI_Nerd_Channel")
    st.markdown("---")

    with st.expander("How this works"):
        st.write("""
This app lets you run an A/B test on two value propositions using synthetic respondents (personas) generated with an LLM.
**Caveat:** Results are *synthetic* and meant for early-stage concept testing only. They are not representative of real populations.
""")

    # Sidebar configuration
    st.sidebar.header("LLM Configuration")
    
    # API Key input (Groq only)
    api_key = st.sidebar.text_input("Groq API Key", type="password")
    
    # Model selection (all via Groq)
    model_options = [
        "openai/gpt-oss-20b",  # OpenAI model via Groq
        "llama3-70b-8192",
        "llama3-8b-8192"
    ]
    
    model_name = st.sidebar.selectbox(
        "Groq Model", 
        model_options,
        index=0,  # Default to openai/gpt-oss-20b
        help="Select the Groq model to use (includes OpenAI models via Groq)"
    )
    
    temperature = st.sidebar.slider("LLM temperature", 0.0, 1.0, AppConfig.temperature, 0.05)
    max_tokens = st.sidebar.slider("Max tokens per response", 256, 4096, AppConfig.max_tokens, 64)
    seed = st.sidebar.number_input("Random seed", min_value=0, value=AppConfig.seed, step=1)
    set_seed(int(seed))

    st.sidebar.header("Response Mode")
    response_mode = st.sidebar.radio("Select response mode", ["Likert", "Binary (Yes/No)"])

    if response_mode == "Likert":
        scale_max = st.sidebar.selectbox("Likert scale max", [5, 7, 10], index=0)
    else:
        scale_max = None

    st.header("Inputs")

    col1, col2, col3 = st.columns(3)
    with col1:
        age_min = st.number_input("Minimum age", min_value=13, max_value=120, value=18, step=1)
    with col2:
        age_max = st.number_input("Maximum age", min_value=13, max_value=120, value=65, step=1)
    with col3:
        geo_focus = st.selectbox("Geographical focus", ["Germany", "Europe", "Global"])

    target_segment = st.text_area("Target customer segment description", height=80,
                                  placeholder="e.g., Early-career professionals in tech startups responsible for tooling decisions.")
    n_resp = st.number_input("Number of respondents (personas)", min_value=5, max_value=500, value=50, step=5)

    st.subheader("Value Proposition Scenarios")
    scenario_a = st.text_area("Scenario A", height=120,
                              placeholder="Describe value proposition A in clear terms.")
    scenario_b = st.text_area("Scenario B", height=120,
                              placeholder="Describe value proposition B in clear terms.")

    st.subheader("Evaluation Question")
    question = st.text_area("Question to ask each persona",
                            placeholder="e.g., How likely are you to try this product in the next 30 days?")

    run = st.button("Run A/B experiment")

    if not run:
        st.stop()

    # Basic validation
    if age_max < age_min:
        st.error("Maximum age must be greater than or equal to minimum age.")
        st.stop()

    if not target_segment.strip():
        st.error("Please provide a target customer segment description.")
        st.stop()
    if not scenario_a.strip() or not scenario_b.strip():
        st.error("Please provide both Scenario A and Scenario B.")
        st.stop()
    if not question.strip():
        st.error("Please provide the evaluation question.")
        st.stop()

    # Step 1 — Create synthetic sample
    st.markdown("### Step 1 — Create random sample")
    with st.spinner("Sampling respondents..."):
        df = sample_respondents(int(n_resp), int(age_min), int(age_max), geo_focus)
        df["age_group"] = df["age"].apply(assign_age_group)
    st.dataframe(df.head(10), use_container_width=True)
    st.success(f"Created a synthetic sample with {len(df)} respondents.")

    # Step 2 & 3 — Persona description and answers for both scenarios
    st.markdown("### Step 2 — Generate persona + answers for A and B")
    
    # Validate API key and package availability
    if not api_key:
        st.error("Groq API Key is required.")
        st.stop()
    
    if Groq is None:
        st.error("The 'groq' package is not available. Install it to use Groq models.")
        st.stop()

    personas: List[str] = []
    ansA_text: List[str] = []
    ansB_text: List[str] = []
    A_numeric: List[Optional[float]] = []
    B_numeric: List[Optional[float]] = []

    progress = st.progress(0)
    for i, row in df.iterrows():
        profile = f"Age: {row['age']}; Gender: {row['gender']}; Nationality: {row['nationality']}."
        messages = persona_prompt(
            profile=profile,
            target_segment=target_segment,
            question=question,
            scenario_a=scenario_a,
            scenario_b=scenario_b,
            response_mode="Likert" if response_mode == "Likert" else "Binary",
            scale_max=int(scale_max) if scale_max else 5,
        )
        try:
            raw = call_groq(api_key, model_name, messages, temperature, max_tokens)
            parsed = parse_persona_json(raw)
            if not parsed:
                # attempt one lightweight retry with lower temperature
                raw = call_groq(api_key, model_name, messages, max(0.0, temperature - 0.1), max_tokens)
                parsed = parse_persona_json(raw)
            if not parsed:
                raise ValueError("Model did not return valid JSON.")
            personas.append(parsed.get("persona", "").strip())

            if response_mode == "Likert":
                sa = parsed.get("scenario_A", {})
                sb = parsed.get("scenario_B", {})
                a_rating = sa.get("rating", None)
                b_rating = sb.get("rating", None)
                ansA_text.append(str(sa.get("answer_text", "")).strip())
                ansB_text.append(str(sb.get("answer_text", "")).strip())
                try:
                    A_numeric.append(int(a_rating))
                except Exception:
                    A_numeric.append(None)
                try:
                    B_numeric.append(int(b_rating))
                except Exception:
                    B_numeric.append(None)
            else:
                sa = parsed.get("scenario_A", {})
                sb = parsed.get("scenario_B", {})
                a_choice = str(sa.get("choice", "")).strip()
                b_choice = str(sb.get("choice", "")).strip()
                ansA_text.append(str(sa.get("answer_text", "")).strip() or a_choice)
                ansB_text.append(str(sb.get("answer_text", "")).strip() or b_choice)
                A_numeric.append(1 if a_choice.lower() == "yes" else (0 if a_choice.lower() == "no" else None))
                B_numeric.append(1 if b_choice.lower() == "yes" else (0 if b_choice.lower() == "no" else None))
        except Exception as e:
            personas.append("ERROR: " + str(e))
            ansA_text.append("")
            ansB_text.append("")
            A_numeric.append(None)
            B_numeric.append(None)

        progress.progress(int((i + 1) / len(df) * 100))

    df["persona"] = personas
    df["A_answer_text"] = ansA_text
    df["B_answer_text"] = ansB_text
    df["A_value"] = A_numeric
    df["B_value"] = B_numeric

    st.dataframe(df.head(10), use_container_width=True)
    st.success("Step 2 complete.")

    # Step 4 — Statistical Analysis
    st.markdown("### Step 3 — Statistical analysis")

    valid = df.dropna(subset=["A_value", "B_value"]).copy()
    n_valid = len(valid)
    if n_valid < 5:
        st.warning("Fewer than 5 valid paired responses. Statistical power will be very low.")

    if response_mode == "Likert":
        a = valid["A_value"].to_numpy(dtype=float)
        b = valid["B_value"].to_numpy(dtype=float)
        diff = float(np.mean(b - a))
        p_perm = paired_permutation_test(a, b, n_perms=int(AppConfig.n_permutations), seed=int(seed))
        ci_lo, ci_hi = bootstrap_ci_mean_diff(a, b, n_boot=int(AppConfig.n_bootstrap), seed=int(seed))
        d_z = cohen_dz(a, b)

        st.write(f"**Overall (n={n_valid})**")
        st.write(f"Mean A = {np.mean(a):.2f}, Mean B = {np.mean(b):.2f}, Mean (B - A) = {diff:.2f}")
        st.write(f"Permutation test p-value (two-sided) = {p_perm:.4f}")
        st.write(f"Bootstrap 95% CI for mean difference = [{ci_lo:.2f}, {ci_hi:.2f}]")
        st.write(f"Cohen's d (paired) = {d_z:.2f}")

        plot_mean_bars(["A", "B"], [float(np.mean(a)), float(np.mean(b))],
                       "Mean ratings by scenario", "Mean rating")

        # Gender splits
        st.subheader("By gender")
        for g, sub in valid.groupby("gender"):
            a_g = sub["A_value"].to_numpy(dtype=float)
            b_g = sub["B_value"].to_numpy(dtype=float)
            if len(sub) >= 5:
                diff_g = float(np.mean(b_g - a_g))
                p_g = paired_permutation_test(a_g, b_g, n_perms=2000, seed=int(seed))
                st.write(f"{g} (n={len(sub)}): Mean(B-A)={diff_g:.2f}, p={p_g:.4f}")
            else:
                st.write(f"{g} (n={len(sub)}): sample too small for significance testing.")
        # Age splits
        st.subheader("By age group")
        for ag, sub in valid.groupby("age_group"):
            a_g = sub["A_value"].to_numpy(dtype=float)
            b_g = sub["B_value"].to_numpy(dtype=float)
            if len(sub) >= 5:
                diff_g = float(np.mean(b_g - a_g))
                p_g = paired_permutation_test(a_g, b_g, n_perms=2000, seed=int(seed))
                st.write(f"{ag} (n={len(sub)}): Mean(B-A)={diff_g:.2f}, p={p_g:.4f}")
            else:
                st.write(f"{ag} (n={len(sub)}): sample too small for significance testing.")

    else:
        # Binary Yes/No -> McNemar
        # Build paired counts
        mapping = {1: "Yes", 0: "No"}
        a = valid["A_value"].astype(int).to_numpy()
        b = valid["B_value"].astype(int).to_numpy()
        a_yes_b_no = int(np.sum((a == 1) & (b == 0)))
        a_no_b_yes = int(np.sum((a == 0) & (b == 1)))
        a_yes_b_yes = int(np.sum((a == 1) & (b == 1)))
        a_no_b_no = int(np.sum((a == 0) & (b == 0)))

        p_mcnemar = mcnemar_exact(a_no_b_yes, a_yes_b_no)

        st.write(f"**Overall (n={n_valid})**")
        st.write(f"Pairs: A=Yes,B=Yes: {a_yes_b_yes}; A=Yes,B=No: {a_yes_b_no}; A=No,B=Yes: {a_no_b_yes}; A=No,B=No: {a_no_b_no}")
        st.write(f"McNemar exact p-value (two-sided) = {p_mcnemar:.4f}")
        prop_a = float(np.mean(a))
        prop_b = float(np.mean(b))
        st.write(f"Proportion Yes — A: {prop_a:.2f}, B: {prop_b:.2f}, Δ = {prop_b - prop_a:.2f}")

        plot_mean_bars(["A (Yes=1)", "B (Yes=1)"], [prop_a, prop_b],
                       "Proportion 'Yes' by scenario", "Proportion Yes")

        st.subheader("By gender")
        for g, sub in valid.groupby("gender"):
            if len(sub) >= 5:
                a = sub["A_value"].astype(int).to_numpy()
                b = sub["B_value"].astype(int).to_numpy()
                a_yes_b_no = int(np.sum((a == 1) & (b == 0)))
                a_no_b_yes = int(np.sum((a == 0) & (b == 1)))
                p = mcnemar_exact(a_no_b_yes, a_yes_b_no)
                st.write(f"{g} (n={len(sub)}): McNemar p={p:.4f}")
            else:
                st.write(f"{g} (n={len(sub)}): sample too small for significance testing.")

        st.subheader("By age group")
        for ag, sub in valid.groupby("age_group"):
            if len(sub) >= 5:
                a = sub["A_value"].astype(int).to_numpy()
                b = sub["B_value"].astype(int).to_numpy()
                a_yes_b_no = int(np.sum((a == 1) & (b == 0)))
                a_no_b_yes = int(np.sum((a == 0) & (b == 1)))
                p = mcnemar_exact(a_no_b_yes, a_yes_b_no)
                st.write(f"{ag} (n={len(sub)}): McNemar p={p:.4f}")
            else:
                st.write(f"{ag} (n={len(sub)}): sample too small for significance testing.")

    # Downloads
    st.markdown("### Download results")
    out = df.copy()
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, file_name="ab_value_prop_results.csv", mime="text/csv")

    st.markdown("---")
    st.caption("NOTE: This app uses synthetic respondents (LLM-generated personas) and should be used for early concept validation only.")
    st.caption("If you need fully representative results, run a real survey with an appropriate sample frame and weighting.")

if __name__ == "__main__":
    main()
