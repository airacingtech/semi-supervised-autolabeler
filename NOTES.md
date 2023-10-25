# Useful commands 

### RabbitMQ
http://label.roarart.online:15672/

rabbitmqctl add_user username password
rabbitmqctl set_user_tags username administrator

### Celery
celery -A roar_server.celery worker --loglevel=info -P eventlet -E -n worker1
celery -A roar_server.celery worker --loglevel=info -P eventlet -E -n worker2

pkill -9 -f 'celery worker'

### Flask
python roar_server.py
OR
flask --app roar_server --debug run