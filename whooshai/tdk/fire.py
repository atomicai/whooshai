import random
import time

from celery import Celery
from kombu import Connection, Exchange, Queue
from kombu.mixins import ConsumerMixin


class Worker(ConsumerMixin):
    def __init__(self, connection, queues):
        self.connection = connection
        self.queues = queues

    def get_consumers(self, Consumer, channel):
        return [Consumer(queues=self.queues, callbacks=[self.on_message])]

    def on_message(self, body, message):
        print('Got message: {0}'.format(body))
        message.ack()


flow = Celery('fire', broker='amqp://justatom:fate@localhost:5672', backend="redis://localhost:6379/0")


@flow.task
def up():
    # TODO: Load the model(s)
    return 1


@flow.task
def infer(data):
    # TODO: Perform inference
    return 1


@flow.task(bind=True)
def jobber(self):
    """Background task that runs a long function with progress reports."""
    verb = ['Starting up', 'Booting', 'Repairing', 'Loading', 'Checking']
    adjective = ['master', 'radiant', 'silent', 'harmonic', 'fast']
    noun = ['solar array', 'particle reshaper', 'cosmic ray', 'orbiter', 'bit']
    message = ''
    total = 22
    for i in range(total):
        if not message or random.random() < 0.25:
            message = '{0} {1} {2}...'.format(random.choice(verb), random.choice(adjective), random.choice(noun))
        self.update_state(state='PROGRESS', meta={'current': i, 'total': total, 'status': message})
        time.sleep(1)
    return {'current': total, 'total': total, 'status': 'Task completed!', 'result': 42}


@flow.task
def fetcher():
    exchange = Exchange("", type="direct")
    queues = [Queue("AIResponse", exchange)]
    rabbit_url = "amqp://justatom:fate@localhost:5672/"

    with Connection(rabbit_url, heartbeat=4) as conn:
        worker = Worker(conn, queues)
        worker.run()
