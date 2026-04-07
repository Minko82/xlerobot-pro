FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 8501

CMD ["streamlit", "run", "src/xlerobot_soft_orchestrator/ui.py", "--server.address=0.0.0.0", "--server.port=8501"]

