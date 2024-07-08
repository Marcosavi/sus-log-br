# build_files.sh
pip install -r requirements.txt
python -m venv venv
python3.9 manage.py collectstatic