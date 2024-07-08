#!/bin/bash

# Ensure the correct paths are set
export PATH=$PATH:/vercel/path0/.python/bin:/vercel/path0/bin

# Install dependencies
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt

# Collect static files
python3 manage.py collectstatic --noinput

# Apply database migrations
python3 manage.py migrate
