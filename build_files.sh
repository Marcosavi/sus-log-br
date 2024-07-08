#!/bin/bash

# Add the directory to PATH
export PATH=$PATH:/python312/bin

# Install dependencies
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --noinput

# Apply database migrations
python manage.py migrate