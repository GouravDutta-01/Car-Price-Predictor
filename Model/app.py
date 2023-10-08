from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS, cross_origin

app = Flask(__name__)


CORS(app, supports_credentials=True)

model = pickle.load(open('model/model.pkl', 'rb'))


@app.route('/')
def home():
    return jsonify({'Message': 'API is Running'})


@app.route('/predict', methods=['POST'])
@cross_origin(origin='http://localhost:3000', methods=['POST'],
              allow_headers=['Content-Type'])
def predict():
    try:
        data = request.get_json()
        query_df = pd.DataFrame([data])
        prediction = model.predict(query_df)
        return jsonify({'Prediction': list(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
