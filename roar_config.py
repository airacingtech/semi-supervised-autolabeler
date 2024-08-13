import os
PORT = 5000
HOST = "localhost"
DOWNLOADS_PATH = "/home/roar-perception/cvat_docker/downloads"
PORT = 5000
FLASK_APP="roar_server.py"
FLASK_ENV="dev"
DEBUG = True

RMQ_USER=""
RMQ_PW=""
DB_URL="sqlite:///jobs.db"

if DOWNLOADS_PATH == ...:
    raise ValueError("DOWNLOADS_PATH must be set in roar_config.py")

if os.path.exists(DOWNLOADS_PATH) is False:
    raise ValueError("DOWNLOADS_PATH must exist")

# Expand ~ to the user's home directory
DOWNLOADS_PATH = os.path.expanduser(DOWNLOADS_PATH)

CVAT_PATH = os.path.join(DOWNLOADS_PATH, 'updates.txt')
