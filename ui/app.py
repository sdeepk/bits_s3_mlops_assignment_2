import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:9000/predict"
METRICS_URL = "http://localhost:9000/metrics"

st.set_page_config(page_title="Cats vs Dogs AI", layout="centered")

# ---- Styling ----
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .stButton>button {
        background-color: #121212;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🐾 Cats vs Dogs AI Classifier")
st.write("Upload an image and let the model decide!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ---- Prediction Section ----
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("🔍 Predict"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Animated fake loading
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)

        status_text.text("Sending image to model...")

        response = requests.post(
            API_URL,
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
        )

        progress_bar.empty()
        status_text.empty()

        if response.status_code == 200:
            result = response.json()
            label = result["label"]
            confidence = result["probability"]

            # ---- Emoji Reaction ----
            if label.lower() == "cat":
                st.success("🐱 It's a Cat!")
            else:
                st.success("🐶 It's a Dog!")

            # ---- Confidence Gauge ----
            st.subheader("Confidence Level")
            st.progress(float(confidence))

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence*100:.2f}%"
            )

        else:
            st.error("Prediction failed")

st.markdown("---")

# ---- Metrics Section ----
st.subheader("📊 Live API Metrics")

if st.button("Refresh Metrics"):
    try:
        metrics_response = requests.get(METRICS_URL)
        metrics_text = metrics_response.text

        # Extract basic metrics
        lines = metrics_text.split("\n")
        request_count = 0
        for line in lines:
            if line.startswith("http_requests_total"):
                request_count = float(line.split(" ")[1])

        # Simulate latency data for visualization
        latency_samples = np.random.normal(0.05, 0.01, 20)
        df = pd.DataFrame({
            "Latency (seconds)": latency_samples
        })

        st.metric("Total Requests", int(request_count))
        st.line_chart(df)

    except Exception:
        st.error("Unable to fetch metrics")
