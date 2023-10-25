rabbitmq - http://label.roarart.online:15672/

rabbitmqctl add_user username password
rabbitmqctl set_user_tags username administrator

celery -A roar_server.celery worker --loglevel=info -P eventlet -E 
# can use gevent / eventlet
# you can set multiple workers by doing --concurrency=2
# but recommended to have 1 worker per machine. question is, are there more machines we can use?

pkill -9 -f 'celery worker'

celery -A roar_server.celery worker --loglevel=info -P eventlet -E -n worker1
celery -A roar_server.celery worker --loglevel=info -P eventlet -E -n worker2