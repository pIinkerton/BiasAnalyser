import re
import pandas as pd
import json
import streamlit as st
from google import genai
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import yt_dlp


# -----------------------------
# Step 1. Load dataset
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
# Step 2. Transcript extractor
# -----------------------------
def extract_transcript(video_url: str) -> str:
    try:
        if "watch?v=" in video_url:
            video_id = video_url.split("watch?v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            raise ValueError("Invalid YouTube URL.")

        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        try:
            transcript = transcript_list.find_manually_created_transcript(['en']).fetch()
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(['en']).fetch()

        return " ".join([seg.text for seg in transcript])

    except Exception:
        # fallback yt-dlp
        try:
            ydl_opts = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en"],
                "subtitlesformat": "json3",
                "quiet": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)

            captions = info.get("subtitles", {}).get("en") or info.get("automatic_captions", {}).get("en")

            if not captions:
                return "❌ No transcript found."

            import requests
            for c in captions:
                if c["ext"] == "json3":
                    data = requests.get(c["url"]).json()
                    lines = []
                    for ev in data.get("events", []):
                        if "segs" in ev:
                            chunk = " ".join(seg.get("utf8", "") for seg in ev["segs"])
                            if chunk.strip():
                                lines.append(chunk.strip())
                    return " ".join(lines)

            return "❌ Transcript format not supported."

        except Exception as e:
            return f"❌ Extraction failed: {str(e)}"


# -----------------------------
# Step 3. Gemini prediction (NEW SDK)
# -----------------------------
def predict_bias_gemini(client, transcript: str, df: pd.DataFrame, k: int = 3) -> dict:
    def score_to_class(score):
        return {
            0.0: "Left",
            0.25: "Left-leaning",
            0.5: "Neutral",
            0.75: "Right-leaning",
            1.0: "Right",
        }.get(score, "Unknown")

    examples = df.sample(min(k, len(df)), random_state=42)

    example_texts = "\n".join(
        f'Example: "{row.transcript[:300]}" → Class: {score_to_class(row.bias_score)}'
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
- Provide a highly reputable source to back up your findings, if relevant or available.
- Format your response as: {{"class": "...", "score": ..., "reason": "..."}}.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        text = response.text.strip()
        return json.loads(text)

    except Exception as e:
        return {"class": "Error", "score": None, "reason": str(e)}


# -----------------------------
# Step 4. Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="Bias Analyser")

    st.title("Bias Analyser")
    st.markdown("Analyse political leaning using Gemini.")

    # Sidebar
    st.sidebar.header("Configuration")

    api_key = st.sidebar.text_input(
        "Enter your Gemini API Key",
        value="AIzaSyDgAv-1U0u1rUalWaZLAOejjBEWg7WLy88",  # you can pre-fill here if needed
        type="password"
    )

    dataset_path = st.sidebar.text_input("Dataset File", "transcript output.txt")

    if not api_key:
        st.warning("Please enter your API key.")
        return

    # ✅ NEW SDK client
    client = genai.Client(api_key=api_key)

    # Load dataset
    try:
        df = load_dataset(dataset_path)
        st.sidebar.success(f"Loaded {len(df)} transcripts")
    except Exception as e:
        st.error(f"Dataset error: {e}")
        return

    # Input
    st.subheader("Input Options")

    youtube_url = st.text_input("YouTube URL (optional)")

    # ✅ Initialize session state BEFORE widget
    if "transcript_input" not in st.session_state:
        st.session_state.transcript_input = ""

    # ✅ Callback function (safe state update)
    def fetch_and_fill():
        if youtube_url:
            transcript = extract_transcript(youtube_url)
            if transcript.startswith("❌"):
                st.error(transcript)
            else:
                st.session_state.transcript_input = transcript
                st.success("Transcript loaded")

    # ✅ Button uses callback
    st.button("Fetch Transcript", on_click=fetch_and_fill)

    # ✅ Widget reads from session state
    st.text_area("Transcript", key="transcript_input", height=200)

    # Classify
    if st.button("Classify"):
        transcript = st.session_state["transcript_input"].strip()

        if not transcript:
            st.warning("Enter transcript first.")
            return

        with st.spinner("Analysing..."):
            result = predict_bias_gemini(client, transcript, df)

        st.subheader("Result")
        st.write(f"**Bias:** {result.get('class')}")
        st.write(f"**Reason:** {result.get('reason')}")


if __name__ == "__main__":
    main()
