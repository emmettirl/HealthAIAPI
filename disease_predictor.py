from random import sample, randint
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


class DiseasePredictor:
    def __init__(self, filepath):
        self.random_forest_model = None
        self.label_encoder = None
        self.one_hot_encoded_df_cleaned = None

        # Load the dataset and clean data
        self.dataset = pd.read_csv(filepath)
        self.clean_data()

        # Split data and train the model
        X_train, X_test, y_train, y_test = self.split_data()
        self.train_model(X_train, y_train)
        self.check_accuracy(X_train, X_test, y_train, y_test)



    def clean_data(self):
        # fill empty cells
        dataset_filled = self.dataset.fillna('No Symptom')
        symptoms_combined = dataset_filled.iloc[:, 1:].apply(lambda x: x[x != 'No Symptom'].dropna().tolist(), axis=1)


        mlb = MultiLabelBinarizer()
        one_hot_encoded_symptoms = mlb.fit_transform(symptoms_combined)
        symptom_columns = mlb.classes_

        cleaned_symptom_columns = [symptom.strip().lower() for symptom in symptom_columns]

        column_mapping = {original: cleaned for original, cleaned in zip(symptom_columns, cleaned_symptom_columns)}
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded_symptoms, columns=symptom_columns)

        self.one_hot_encoded_df_cleaned = one_hot_encoded_df.rename(columns=column_mapping)
        self.dataset = pd.concat([dataset_filled['Disease'], self.one_hot_encoded_df_cleaned], axis=1).drop_duplicates()

    def split_data(self):
        X = self.dataset.drop('Disease', axis=1)
        y = self.dataset['Disease']
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        return train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        self.random_forest_model = RandomForestClassifier(random_state=42, max_depth=15)
        self.random_forest_model.fit(X_train, y_train)

    def check_accuracy(self, X_train, X_test, y_train, y_test):
        # Print training and test accuracy
        print("Training Accuracy:", self.random_forest_model.score(X_train, y_train))
        print("Test Accuracy:", self.random_forest_model.score(X_test, y_test))

    def predict_disease(self, presented_symptoms):
        # pass a list of diseases, return a diagnosis.
        all_symptoms = self.one_hot_encoded_df_cleaned.columns.tolist()
        input_vector = self.create_input_vector(presented_symptoms, all_symptoms)
        input_df = pd.DataFrame([input_vector], columns=all_symptoms)
        predicted_label = self.random_forest_model.predict(input_df)
        predicted_disease = self.label_encoder.inverse_transform(predicted_label)
        return predicted_disease[0]

    @staticmethod
    def create_input_vector(symptoms, all_symptoms): # used to convert list into ML model readable input
        input_vector = [0] * len(all_symptoms)
        for symptom in symptoms:
            if symptom in all_symptoms:
                index = all_symptoms.index(symptom)
                input_vector[index] = 1
        return input_vector

    @staticmethod # Use this to get a list of symptoms, to be reflected in app for selection
    def get_symptom_labels(encoded_df):
        return encoded_df.columns.tolist()

    def model_test_random_symptoms(self):
        # test function, generates a random list of symptoms, and feeds it to the predictor.
        symptom_labels = self.get_symptom_labels(self.one_hot_encoded_df_cleaned)
        random_subset = sample(symptom_labels, randint(1, 5))
        print(type(random_subset))
        prediction = self.predict_disease(random_subset)
        return random_subset, prediction


def main():
    dataset = "data/dataset.csv"
    random_forest_model = DiseasePredictor(dataset)

    # Test the model with random symptoms
    print(random_forest_model.model_test_random_symptoms())

if __name__ == '__main__':
    main()
