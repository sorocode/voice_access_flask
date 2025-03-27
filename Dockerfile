# Python 3.9 이미지를 기반으로 설정
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 파일을 복사
COPY requirements.txt .

# requirements.txt에서 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# Gunicorn으로 Flask 애플리케이션 실행
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5001"]