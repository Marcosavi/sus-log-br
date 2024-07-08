#!/bin/bash

# Ensure the correct paths are set
export PATH=$PATH:/vercel/path0/.python/bin

# Install dependencies
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --noinput

# Apply database migrations
python manage.py migrate
