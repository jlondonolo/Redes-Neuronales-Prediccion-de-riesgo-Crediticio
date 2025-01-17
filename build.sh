#!/usr/bin/env bash
# exit on error
set -o errexit

# Crear directorio faltante y establecer permisos
mkdir -p /var/lib/apt/lists/partial
chmod -R 644 /var/lib/apt/lists/partial

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
