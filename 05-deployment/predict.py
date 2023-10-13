from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)


with open('./dv.bin', 'rb') as dv:
    dv = pickle.load(dv)

with open('./model1.bin', 'rb') as model:
    model = pickle.load(model)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    client_data = dv.transform(data)

    prediction = model.predict_proba(client_data)
    return jsonify({'prediction': prediction[0, 1]})


if __name__ == '__main__':
    app.run(debug=True)
