import os
PORT = 5000
HOST = "localhost"
DOWNLOADS_PATH = "/home/roar-go/cvat_docker/downloads"
CVAT_PATH = os.path.join(DOWNLOADS_PATH, 'updates.txt')
UPLOAD_FOLDER = DOWNLOADS_PATH
PORT = 5000
FLASK_APP="roar_server.py"
FLASK_ENV="dev"
DEBUG = True



RMQ_USER=""
RMQ_PW=""

DB_URL="sqlite:///jobs.db"

