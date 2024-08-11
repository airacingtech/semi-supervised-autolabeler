import os
PORT = 5000
HOST = "localhost"
DOWNLOADS_PATH = ...
UPLOAD_FOLDER = ...
PORT = 5000
FLASK_APP="roar_server.py"
FLASK_ENV="dev"
DEBUG = True



RMQ_USER=""
RMQ_PW=""

DB_URL="sqlite:///jobs.db"

if DOWNLOADS_PATH == ...:
    raise ValueError("DOWNLOADS_PATH must be set in roar_config.py")

if UPLOAD_FOLDER == ...:
    raise ValueError("UPLOAD_FOLDER must be set in roar_config.py")

CVAT_PATH = os.path.join(DOWNLOADS_PATH, 'updates.txt')
