FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir "streamlit>=1.57" httpx

COPY app ./app

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
