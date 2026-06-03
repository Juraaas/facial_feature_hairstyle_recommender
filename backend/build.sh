#!/bin/bash
apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libgles2-mesa \
    libegl1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    --no-install-recommends

pip install -r requirements.txt