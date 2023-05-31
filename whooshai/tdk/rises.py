import os
import random
import time
from pathlib import Path
import os
from flask import Flask, request, render_template, session, flash, redirect, \
    url_for, jsonify
from celery import Celery


app = Flask(
    __name__,
    template_folder="build",
    static_folder="build",
    root_path=Path(os.getcwd()) / "whoosh",
)

app.config['SECRET_KEY'] = 'deshibasara'


# Celery configuration
app.config['CELERY_BROKER_URL'] = 'pyamqp://justatom:fate@localhost:5672'
# app.config["CELERY_TASK_RESULT_EXPIRES"] = 18000
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
# app.config["CELERY_RESULT_EXCHANGE"] = "celery_result"


# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'], worker_state_db = '/tmp/celery_state')
celery.conf.update(app.config)



@celery.task(bind=True)
def long_task(self):
    """Background task that runs a long function with progress reports."""
    verb = ['Starting up', 'Booting', 'Repairing', 'Loading', 'Checking']
    adjective = ['master', 'radiant', 'silent', 'harmonic', 'fast']
    noun = ['solar array', 'particle reshaper', 'cosmic ray', 'orbiter', 'bit']
    message = ''
    total = 22
    for i in range(total):
        # if not message or random.random() < 0.25:
        #     message = '{0} {1} {2}...'.format(random.choice(verb),
        #                                       random.choice(adjective),
        #                                       random.choice(noun))
        self.update_state(state='PROGRESS',
                          meta={'current': i, 'total': total,
                                'status': message})
        time.sleep(1)
    return {'current': total, 'total': total, 'status': 'Task completed!',
            'result': 42}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', email=session.get('email', ''))
    return redirect(url_for('index'))


@app.route('/longtask', methods=['POST'])
def longtask():
    task = long_task.apply_async()
    return jsonify({}), 202, {'Location': url_for('taskstatus',
                                                  task_id=task.id)}


@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = long_task.AsyncResult(task_id)
    print(task.state)
    # print(task.info.get("current", "CURRENT undefined"))
    # print(task.info.get("total", "TOTAL undefined"))
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
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
    app.run(debug=False, host="0.0.0.0", port=4321)