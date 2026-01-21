FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir streamlit requests

# Copy frontend code
COPY frontend/streamlit_app.py ./streamlit_app.py

# Streamlit runs on port 8080 for Cloud Run
EXPOSE 8080

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
