import streamlit as st
import os
import yt_dlp
import librosa
#from speechbrain.pretrained import EncoderClassifier
from speechbrain.inference.classifiers import EncoderClassifier
import torch
import torchaudio

# Set page configuration
st.set_page_config(page_title="Accent Analyzer", page_icon="🗣️", layout="centered")

# App title and description
st.title("🗣️ English Accent Analyzer")
st.markdown("""
    This tool analyzes the English accents from a public video URL.
    Paste a link from YouTube, Loom or a direct MP4 link to get started.
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
    """Analyzes the accent from an audio file and returns the top 3 predictions."""
    if not classifier:
        st.error("Model not loaded. Cannot analyze accent.")
        return None

    # This is the list of 16 accents the model was trained on, in order.
    # We get this from the model's label_encoder.txt file on Hugging Face.
    accent_labels = [
        'African', 'Australian', 'Canadian', 'England', 'Hongkong', 'Indian',
        'Irish', 'Malaysian', 'Newzealand', 'Northernireland', 'Philippines',
        'Scottish', 'Singapore', 'Southatlandtic', 'US', 'Welsh'
    ]

    try:
        signal, fs = torchaudio.load(audio_file)

        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)

        # Get the raw output probabilities from the model
        out_prob, _, _, _ = classifier.classify_batch(signal)

        # Apply softmax to get confidence scores as percentages
        scores = F.softmax(out_prob, dim=-1) * 100

        # Squeeze the tensor to remove unnecessary dimensions and convert to a simple list
        scores_list = scores[0].tolist()

        # Pair each label with its score
        results = []
        for label, score in zip(accent_labels, scores_list):
            results.append({"accent": label, "confidence": score})

        # Sort the results by confidence in descending order
        sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)

        # Return the top 3
        return sorted_results[:3]

    except Exception as e:
        st.error(f"Error analyzing accent: {e}")
        return None
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
                # MODIFIED UI TO DISPLAY TOP 3 RESULTS
                top_results = analyze_accent(audio_file)
                if top_results:
                    st.success("Analysis Complete!")
                    st.subheader("Top 3 Accent Predictions:")

                    # Create columns for a cleaner layout
                    cols = st.columns(3)
                    for i, result in enumerate(top_results):
                        with cols[i]:
                            st.metric(
                                label=f"#{i+1} Accent",
                                value=result['accent']
                            )
                            st.metric(
                                label="Confidence",
                                value=f"{result['confidence']:.2f}%"
                            )
    else:
        st.warning("Please enter a video URL.")
# Add a footer
st.markdown("---")
st.markdown("Developed by Jawad")