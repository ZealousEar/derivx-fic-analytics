FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY setup.py .
COPY pyproject.toml .
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create Streamlit config
RUN mkdir -p ~/.streamlit && \
    echo '[server]\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\n' > ~/.streamlit/config.toml

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app
ENTRYPOINT ["streamlit", "run", "src/derivx/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
