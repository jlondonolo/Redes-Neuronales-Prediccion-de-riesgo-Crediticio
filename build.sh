#!/usr/bin/env bash
# exit on error
set -o errexit

# Instalar dependencias del sistema
apt-get update
apt-get install -y \
    build-essential \
    python3-dev \
    gfortran \
    libopenblas-dev

# Instalar dependencias de Python
pip install --upgrade pip
pip install -r requirements.txt
