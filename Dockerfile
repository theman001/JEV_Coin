FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# 의존성 레이어를 먼저 (소스만 바뀌면 재설치 안 함). aarch64(ARM64) 휠이 있어 컴파일 없이 설치됨 (ta 만 순수 파이썬 sdist)
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src

# 상태/판단 이력은 /data (볼륨). 비루트 사용자로 실행
RUN useradd --create-home app && mkdir /data && chown app:app /data
USER app
ENV DATA_DIR=/data PORT=8787
EXPOSE 8787

CMD ["python", "-m", "src.main"]
