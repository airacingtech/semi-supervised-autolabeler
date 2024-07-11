#!/bin/bash
WORKER_num="$2"
celery -A roar_server.celery worker --loglevel=info -P eventlet -E -n worker${WORKER_num}