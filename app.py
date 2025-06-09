import streamlit as st
import os
import yt_dlp
import librosa
from speechbrain.pretrained import EncoderClassifier
import torch
import torchaudio

# Set page configuration
st.set_page_config(page_title="Accent Analyzer", page_icon="🗣️", layout="centered")

# App title and description
st.title("🗣️ English Accent Analyzer")
st.markdown("""
    This tool analyzes the English accent from a public video URL.
    Paste a link from YouTube, Loom, or a direct MP4 link to get started.
""")

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the pre-trained accent classification model."""
    try:
        classifier = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa",
            savedir="pretrained_models/accent-id-commonaccent_ecapa"
        )
        return classifier
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

classifier = load_model()

# --- Core Functions ---
def download_audio(video_url):
    """Downloads audio from a video URL."""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'downloaded_audio.%(ext)s',
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return "downloaded_audio.mp3"
    except Exception as e:
        st.error(f"Error downloading audio: {e}")
        return None

def analyze_accent(audio_file):
    """Analyzes the accent from an audio file."""
    if not classifier:
        st.error("Model not loaded. Cannot analyze accent.")
        return None, None

    try:
        # Resample the audio to 16kHz as required by the model
        signal, fs = torchaudio.load(audio_file)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)

        # Classify the accent
        out_prob, score, index, text_lab = classifier.classify_batch(signal)
        return text_lab[0], score.item() * 100
    except Exception as e:
        st.error(f"Error analyzing accent: {e}")
        return None, None
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)

# --- Streamlit UI ---
video_url = st.text_input("Enter the public video URL:", placeholder="e.g., https://www.youtube.com/watch?v=...")

if st.button("Analyze Accent"):
    if video_url:
        with st.spinner("Analyzing... This may take a moment."):
            audio_file = download_audio(video_url)
            if audio_file:
                accent, confidence = analyze_accent(audio_file)
                if accent and confidence is not None:
                    st.success("Analysis Complete!")
                    st.metric(label="Detected Accent", value=accent)
                    st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

                    st.markdown("---")
                    st.info(
                        "**Explanation:**\n\n"
                        "- **Detected Accent:** The most likely English accent identified in the audio.\n"
                        "- **Confidence Score:** The model's confidence in its prediction."
                    )
    else:
        st.warning("Please enter a video URL.")

# Add a footer
st.markdown("---")
st.markdown("Developed by Jawad")