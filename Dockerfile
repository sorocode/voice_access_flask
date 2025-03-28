# Python 3.12 기반 이미지
FROM python:3.12-slim

# FFmpeg 설치를 위해 apt 업데이트
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY . .

# audio, models 폴더 생성
RUN mkdir -p /app/audio /app/models

# Gunicorn 실행
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5001"]