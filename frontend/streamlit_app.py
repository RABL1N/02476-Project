import streamlit as st
import requests
from typing import Optional

st.set_page_config(page_title="Pneumonia Detection", layout="centered")

# Configuration
DEFAULT_API_URL = "https://mlops-fastapi-304008424690.europe-west1.run.app"
REQUEST_TIMEOUT = 30

# Sidebar for API configuration
with st.sidebar:
    st.header("Configuration")
    api_base_url = st.text_input(
        "API Base URL",
        value=DEFAULT_API_URL,
        help="Base URL of your FastAPI backend (without /predict endpoint)",
    )
    api_url = f"{api_base_url.rstrip('/')}/predict"

st.title("Chest X-Ray Pneumonia Detection")

st.write("Upload a chest X-ray image and the model will predict " "whether it shows **NORMAL** lungs or **PNEUMONIA**.")

uploaded_file = st.file_uploader(
    "Upload an X-ray image",
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG, JPEG",
)

if uploaded_file is not None:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    with col2:
        st.write("**File Info:**")
        st.write(f"Name: {uploaded_file.name}")
        st.write(f"Size: {len(uploaded_file.getvalue()) / 1024:.2f} KB")
        st.write(f"Type: {uploaded_file.type}")

    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner("Running inference..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                response = requests.post(
                    api_url,
                    files=files,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()

                result = response.json()
                prediction = result.get("prediction", "Unknown")

                if prediction == "NORMAL":
                    st.success(f"Prediction: **{prediction}**")
                elif prediction == "PNEUMONIA":
                    st.warning(f"Prediction: **{prediction}**")
                else:
                    st.info(f"Prediction: **{prediction}**")

            except requests.exceptions.Timeout:
                st.error("Request timed out. The API may be slow or unavailable.")
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not connect to API at {api_url}. "
                    "Please check if the API is running and the URL is correct."
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP Error {e.response.status_code}: {e.response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {str(e)}")
            except KeyError:
                st.error("Invalid response format from API.")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# Health check button
if st.sidebar.button("Test API Connection"):
    health_url = f"{api_base_url.rstrip('/')}/health"
    with st.sidebar:
        with st.spinner("Checking API..."):
            try:
                response = requests.get(health_url, timeout=5)
                response.raise_for_status()
                st.success("API is reachable!")
            except Exception as e:
                st.error(f"API unreachable: {str(e)}")
