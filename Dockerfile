FROM python:3.10-slim

WORKDIR /app

# Added libgomp1 which is highly required for MediaPipe/OpenMP CPU acceleration
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

EXPOSE $PORT

# Adjusted to allow shell expansion for $PORT while increasing timeout limits
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 600 --graceful-timeout 30 app:app