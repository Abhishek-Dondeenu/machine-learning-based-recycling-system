from flask import Flask, request
from controller import detect_controller
from flask_cors import CORS

import time
import os

baseDir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/detect', methods=['POST'])
def detect():
    image = request.files['image_file']
    file_name = str(time.time()) + image.filename
    file_path = os.path.join(baseDir, 'uploads', file_name)
    image.save(file_path)
    result = detect_controller.start_detection(file_path)
    return result


if __name__ == '__main__':
    app.run(port=3000)
