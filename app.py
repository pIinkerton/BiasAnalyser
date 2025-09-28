import re
import pandas as pd
import random
import json
import streamlit as st
import google.generativeai as genai


# -----------------------------
# Step 1. Load and parse dataset
# -----------------------------
def load_dataset(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    pattern = r"Video URL:.*?\(Bias Score = ([0-9.]+)\)\s*(.*?)(?=\nVideo URL:|$)"
    matches = re.findall(pattern, raw_text, re.S)

    data = []
    for score, transcript in matches:
        data.append({"bias_score": float(score), "transcript": transcript.strip()})

    return pd.DataFrame(data)


# -----------------------------
# Step 2. Gemini prediction
# -----------------------------
genai.configure(api_key="AIzaSyALeiCm3uhw7ZCEECd_2eY7kOAXoUcLbDQ")  # <- replace with your key


def predict_bias_gemini(transcript: str, df: pd.DataFrame, k: int = 5) -> dict:
    """Use Gemini to classify transcript bias, referencing dataset examples."""

    def score_to_class(score):
        mapping = {
            0.0: "Left",
            0.25: "Left-leaning",
            0.5: "Neutral",
            0.75: "Right-leaning",
            1.0: "Right",
        }
        return mapping.get(score, "Unknown")

    # Sample k examples for grounding
    examples = df.sample(min(k, len(df)), random_state=42)
    example_texts = "\n".join(
        f'Example: "{row.transcript}" → Class: {score_to_class(row.bias_score)}, Score: {row.bias_score}'
        for _, row in examples.iterrows()
    )

    prompt = f"""
You are an expert in political discourse analysis.
Classify the political leaning of transcripts based on the dataset.

Here are labeled examples from the dataset:
{example_texts}

Now classify the following transcript:
"{transcript}"

Rules:
- Do not invent or speculate about people, events, or context not in the transcript.
- If something is unclear, state "Not specified".
- In the "reason", quote specific phrases from the transcript that support your decision.
- Output valid JSON only, no Markdown, no code fences, no extra text.
- Do not repeat the transcript in your output.
- Format your response as: {{"class": "...", "score": ..., "reason": "..."}}.
"""

    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)

    text = response.text.strip()

    # Strip out accidental ```json fences
    if text.startswith("```"):
        text = text.strip("` \n")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except Exception:
        return {"class": "Unknown", "score": None, "reason": text}


# -----------------------------
# Step 3. Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="Bias Score Analyser", layout="centered")

    st.title("Bias Score Analyser")
    st.markdown("Analyse political leaning in transcripts using Google Gemini, grounded on a dataset.")

    # API key input (hidden in sidebar)
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")
    dataset_path = st.sidebar.text_input("Dataset File", "transcript output.txt")

    if not api_key:
        url = "https://aistudio.google.com/app/apikey"
        st.warning("Please enter your Gemini API Key in the sidebar. To generate an API key, visit [Google AI Studio](%s)" % url)
        return

    genai.configure(api_key=api_key)

    # Load dataset once
    try:
        df = load_dataset(dataset_path)
        st.sidebar.success(f"Loaded {len(df)} transcripts")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    # Input box
    transcript = st.text_area("Enter Transcript:", height=200)

    if st.button("Classify"):
        if not transcript.strip():
            st.warning("Please enter a transcript before classifying.")
        else:
            with st.spinner("Classifying with Gemini..."):
                result = predict_bias_gemini(transcript, df, k=5)

            st.subheader("Result")
            st.write(f"**Bias:** {result.get('class')}")
            st.write(f"**Reason:** {result.get('reason')}")


if __name__ == "__main__":

    main()



