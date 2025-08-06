FROM python:3.13.3-bookworm


WORKDIR /app


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MUJOCO_GL=egl
ENV DISPLAY=""
CMD ["python", "train.py"]