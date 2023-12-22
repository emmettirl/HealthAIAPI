from flask import Flask, jsonify, request

from disease_predictor import DiseasePredictor
app = Flask(__name__)

dataset = "data/dataset.csv"
ml_model = DiseasePredictor(dataset)

@app.route('/')
def hello_world():  # The front page

    symptoms_list, prediction = ml_model.model_test_random_symptoms()

    return f'\n {symptoms_list} \n{prediction}'




@app.route('/sumptomsList', methods=['POST'])
def get_symptom_list_API():
    symptomLabels = ml_model.get_symptom_labels(ml_model.one_hot_encoded_df_cleaned)
    return jsonify({symptomLabels})


@app.route('/predictAI', methods=['POST'])
def predict_disease_API():
    print("Headers:", request.headers)
    print("Raw Data:", request.data)
    print("Data Type:", type(request.data))

    data = request.get_json(force=True)  # Force JSON parsing regardless of content-type header
    print("Parsed Data:", data)
    print("Parsed Data Type:", type(data))

    if not data or 'symptomsList' not in data:
        return jsonify({'error': 'Invalid data format'}), 400

    symptoms_list = data['symptomsList']

    # rest of your code

    if not symptoms_list:
        return jsonify({'error': 'Symptoms list is empty'}), 400

    prediction = ml_model.predict_disease(symptoms_list)
    return jsonify({'prediction': prediction})


@app.route('/hello', methods=['POST']) # An API function
def hello_world_API():
    return jsonify({'message': 'Hello, World!'})

if __name__ == '__main__':
    app.run(debug=True)