FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 \
    ultralytics \
    opencv-python-headless \
    requests \
    python-dotenv

WORKDIR /app

COPY detector.py .

CMD ["python3", "-u", "detector.py"]
