import os
from dotenv import load_dotenv
load_dotenv()

from celery import Celery


def make_celery(app):
    RMQ_USER = os.environ.get("RMQ_USER", "guest")
    RMQ_PW = os.environ.get("RMQ_PW", "guest")
    celery = Celery(app.import_name,
        broker=f"amqp://{RMQ_USER}:{RMQ_PW}@localhost//",
        backend="rpc://",
    )
    celery.conf.update(
        CELERY_RESULT_BACKEND = 'rpc://localhost',
        CELERY_TASK_SERIALIZER = 'json',
        CELERY_RESULT_SERIALIZER = 'json',
        CELERY_ACCEPT_CONTENT=['json'],
        CELERY_ENABLE_UTC = True
    )
    TaskBase = celery.Task
    class ContextTask(TaskBase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    celery.Task = ContextTask
    return celery