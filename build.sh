#!/usr/bin/env bash
# exit on error
set -o errexit

# Instalar dependencias de Python
pip install --upgrade pip
pip install -r requirements.txt
