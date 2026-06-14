FROM python:3.10-slim

WORKDIR /app

# Cài đặt các công cụ hệ thống cần thiết (nếu có thư viện cần compile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN  pip install --no-cache-dir -r requirements.txt

# copy toàn bộ mã nguồn
COPY src/ ./src/
COPY .env ./.env

EXPOSE 8000
EXPOSE 8501