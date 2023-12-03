from flask import Flask, jsonify

from disease_predictor import DiseasePredictor
app = Flask(__name__)

dataset = "data/dataset.csv"
ml_model = DiseasePredictor(dataset)

@app.route('/')
def hello_world():  # The front page

    symptoms_list, prediction = ml_model.model_test_random_symptoms()

    return f'\n {symptoms_list} \n{prediction}'


@app.route('/hello', methods=['GET']) # An API function
def hello_world_API():
    return jsonify({'message': 'Hello, World!'})

if __name__ == '__main__':
    app.run(debug=True)