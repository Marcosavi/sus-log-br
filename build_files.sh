#!/bin/bash

# Install SQLite development libraries
apk add --no-cache sqlite-libs sqlite-dev

# Install any other dependencies you need
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --noinput

# Apply database migrations
python manage.py migrate