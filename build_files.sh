#!/bin/bash

# Ensure /python312/bin is in the PATH
export PATH=$PATH:/python312/bin

# Install SQLite development libraries
apk add --no-cache sqlite-libs sqlite-dev

# Ensure pip is available and install dependencies
python3.12 -m ensurepip
pip3.12 install -r requirements.txt

# Collect static files
python3.12 manage.py collectstatic --noinput

# Apply database migrations
python3.12 manage.py migrate
