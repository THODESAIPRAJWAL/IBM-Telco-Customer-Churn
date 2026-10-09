# Serves the Streamlit dashboard (default) or the FastAPI scoring service.
#   docker build -t telco-churn .
#   docker run -p 8501:8501 telco-churn                                   # dashboard
#   docker run -p 8000:8000 telco-churn uvicorn api.main:app --host 0.0.0.0 --port 8000   # API
# Build after `python run_pipeline.py` so models/ and reports/ contain the trained artefacts.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt

COPY config.yaml pyproject.toml ./
COPY src ./src
COPY app ./app
COPY api ./api
COPY .streamlit ./.streamlit
COPY models ./models
COPY reports ./reports
COPY data/processed ./data/processed

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
