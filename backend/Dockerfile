FROM python:3.11-slim

WORKDIR /app

# System deps for ALE / pygame headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY . .

# IRIS root is mounted as a read-only volume at /iris
ENV IRIS_ROOT=/iris
ENV IRIS_SRC=/iris/src
ENV CHECKPOINT_DIR=/iris/checkpoints
ENV DEFAULT_DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
