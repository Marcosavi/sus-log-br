#!/bin/bash

echo "BUILD START"

# Use the default Python available on Vercel, typically Python 3
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --noinput

echo "BUILD END"

# [optional] Start the application here
# python manage.py runserver
