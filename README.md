# English Accent Analyzer 🗣️

A simple web application that analyzes the English accent from a public video URL. It extracts the audio, processes it through a machine learning model and displays the top 3 most likely accent predictions with their confidence scores.

This project was built to demonstrate the practical application of audio processing and machine learning models in a user-friendly web interface.

## 🚀 Live Demo

You can test the live application here:

**[https://accentclassifierstapp.streamlit.app/](https://accentclassifierstapp.streamlit.app/)**

## 📸 Screenshot

![Application-streamlitaccent](https://github.com/user-attachments/assets/ec100a1e-f981-4fc0-9d21-02118e8fb3d4)

## ✨ Features

  - **Accepts Public Video URLs:** Works with links from YouTube, Loom and direct MP4 URLs.
  - **Audio Extraction:** Automatically downloads and extracts the audio track from the provided video.
  - **AI-Powered Accent Classification:** Uses a pre-trained model from Hugging Face to identify English accents.
  - **Top 3 Predictions:** Displays the three most likely accents from 16 accents as in pretrained model here: https://huggingface.co/Jzuluaga/accent-id-commonaccent_ecapa/blob/main/accent_encoder.txt with their corresponding confidence scores for a more nuanced analysis.
  - **Simple & Clean UI:** Built with Streamlit for a straightforward and interactive user experience.

## 🛠️ Tech Stack

  - **Language:** **Python 3.9**
  - **Web Framework:** **Streamlit**
  - **Machine Learning & Audio Processing:**
      - **SpeechBrain:** For interfacing with the pre-trained model.
      - **PyTorch:** As the core deep learning framework.
      - **Librosa & torchaudio:** For loading and resampling audio files.
  - **Video & Audio Extraction:**
      - **yt-dlp:** For downloading video/audio content from URLs.
    <!-- end list -->
      * **FFmpeg:** As the underlying dependency for audio conversion.
  - **ML Model:**
      - [Jzuluaga/accent-id-commonaccent\_ecapa](https://huggingface.co/Jzuluaga/accent-id-commonaccent_ecapa) hosted on **Hugging Face**.
  - **Deployment:**
      - **Streamlit Community Cloud**

## ⚙️ Local Development Setup

To run this project on your local machine, follow these steps.

### Prerequisites

  - Python 3.9
  - [Conda](https://www.anaconda.com/products/distribution) or a Python virtual environment manager (`venv`).
  - [Git](https://git-scm.com/downloads/).
  - **FFmpeg:** You must have FFmpeg installed on your system and available in your PATH. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html).

### Installation & Running

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/supremeninjacoder/AccentClassifierSTApp.git
    cd AccentClassifierSTApp
    ```

2.  **Create and activate a Python environment:**

    *Using Conda:*

    ```bash
    conda create --name accent-app python=3.9
    conda activate accent-app
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Hugging Face API Token:**
    The application requires a Hugging Face token to avoid being rate-limited when downloading the model.

      - Create a file inside the `.streamlit` directory named `secrets.toml`.
      - Add your token to this file like so:
        ```toml
        # .streamlit/secrets.toml
        HUGGING_FACE_HUB_TOKEN = "hf_YourTokenHere"
        ```

5.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

    The application should now be running locally at `http://localhost:8501`.

## ☁️ Deployment

This application is deployed on **Streamlit Community Cloud**. The deployment is configured by the following files in the repository:

  - `requirements.txt`: Specifies the Python dependencies.
  - `packages.txt`: Specifies system-level packages for the Debian environment (like `ffmpeg` and `cmake`).
  - `runtime.txt`: Specifies the Python version to be used.
  - `.streamlit/config.toml`: Contains Streamlit server configurations, such as disabling the file watcher to ensure compatibility with PyTorch.

Any `git push` to the `main` branch will automatically trigger a redeployment with the latest changes.

## 📂 Project Structure

```
.
├── .streamlit/
│   └── config.toml      # Disables file watcher for deployment stability
├── app.py               # The main Streamlit application script
├── packages.txt         # System-level dependencies for Streamlit Cloud
├── README.md            # This file
├── requirements.txt     # Python package dependencies
└── runtime.txt          # Specifies the Python version for deployment
```

## 🙏 Acknowledgements

  - This project is powered by the **SpeechBrain** toolkit.
  - The accent classification model was trained and provided by **Jzuluaga** on the Hugging Face Hub.
