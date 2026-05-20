FROM python:3.12-slim

WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10

COPY pyproject.toml ./
COPY app ./app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir .

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
