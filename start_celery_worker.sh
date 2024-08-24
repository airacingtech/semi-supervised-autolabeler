#!/bin/bash
WORKER_num="$2"
celery -A roar_server.celery worker --loglevel=info -P eventlet -E -n worker${WORKER_num}
# celery -A roar_server.celery worker --pool=threads --concurrency=10 --loglevel=info -P eventlet -E -n worker${WORKER_num}