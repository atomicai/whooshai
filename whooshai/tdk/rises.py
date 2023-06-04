import os
from pathlib import Path

from gevent import monkey

monkey.patch_all()
# from flask_socketio import SocketIO, emit
from celery import Celery
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from flask_session import Session
from whooshai.tdk import fire

app = Flask(
    __name__,
    template_folder="build",
    static_folder="build",
    root_path=Path(os.getcwd()) / "whooshai",
)

app.config['SECRET_KEY'] = 'deshibasara'

app.config['CELERY_BROKER_URL'] = 'amqp://justatom:fate@localhost:5672'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# socketio = SocketIO(
#     app,
#     message_queue='amqp://justatom:fate@rabbitmq:5672',
#     logger=True,
#     engineio_logger=True,
#     cors_allowed_origins="*",
#     manage_session=False,
# )


def ignite(app):
    _app = Celery(app.import_name, broker=app.config['CELERY_BROKER_URL'])
    _app.conf.update(app.config)
    TaskBase = _app.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    _app.Task = ContextTask
    return _app


igni = ignite(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    return redirect(url_for('index'))


@app.route('/longtask', methods=['POST'])
def longtask():
    task = fire.jobber.apply_async()
    return jsonify({}), 202, {'Location': url_for('taskstatus', task_id=task.id)}


@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = fire.jobber.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {'state': task.state, 'current': 0, 'total': 1, 'status': 'Pending...'}
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', ''),
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=4321)
