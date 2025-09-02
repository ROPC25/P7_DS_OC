# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
path='MODEL/model.joblib'
model = joblib.load(path)

@app.route('/')
def index():
    return "🌸 Iris Binary Classifier API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = int(model.predict(features)[0])
    proba = float(model.predict_proba(features)[0][1])  # probabilité d'être Setosa
    return jsonify({
        'prediction': prediction,
        'probability_setosa': proba
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)