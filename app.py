from flask import Flask, jsonify, request

from disease_predictor import DiseasePredictor
app = Flask(__name__)

dataset = "data/dataset.csv"
ml_model = DiseasePredictor(dataset)

@app.route('/')
def hello_world():  # The front page

    symptoms_list, prediction = ml_model.model_test_random_symptoms()

    return f'\n {symptoms_list} \n{prediction}'




@app.route('/sumptomsList', methods=['GET'])
def get_symptom_list_API():
    symptomLabels = ml_model.get_symptom_labels(ml_model.one_hot_encoded_df_cleaned)
    return jsonify({symptomLabels})


@app.route('/predictAI', methods=['GET'])
# test URL http://127.0.0.1:5000/predictAI?symptom=redness_of_eyes&symptom=restlessness&symptom=runny_nose
# above should show "common cold"
def predict_disease_API():
    symptoms_list = request.args.getlist('symptom')
    prediction = ml_model.predict_disease(symptoms_list)
    return prediction

@app.route('/hello', methods=['GET']) # An API function
def hello_world_API():
    return jsonify({'message': 'Hello, World!'})

if __name__ == '__main__':
    app.run(debug=True)