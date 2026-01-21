import streamlit as st
import requests

# CHANGE THIS to your Cloud Run URL
API_URL = "https://https://mlops-fastapi-304008424690.europe-west1.run.app/predict"

st.set_page_config(page_title="Pneumonia Detection", layout="centered")

st.title("🫁 Chest X-Ray Pneumonia Detection")

st.write(
    "Upload a chest X-ray image and the model will predict "
    "whether it shows **NORMAL** lungs or **PNEUMONIA**."
)

uploaded_file = st.file_uploader(
    "Upload an X-ray image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Running inference..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                response = requests.post(API_URL, files=files)
                response.raise_for_status()

                prediction = response.json()["prediction"]

                st.success(f"🧠 Prediction: **{prediction}**")

            except Exception as e:
                st.error(f"Prediction failed: {e}")




