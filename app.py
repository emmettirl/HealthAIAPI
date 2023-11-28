from flask import Flask, jsonify
import keras

app = Flask(__name__)


@app.route('/')
def hello_world():  # The front page

    return f'Hello World!\nKeras Version {keras.__version__}'


@app.route('/hello', methods=['GET']) # An API function
def hello_world_API():
    return jsonify({'message': 'Hello, World!'})

if __name__ == '__main__':
    app.run(debug=True)