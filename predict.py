import pickle
import xgboost as xgb
import numpy as np
from flask import Flask, request, jsonify

app = Flask('laptop_price_predictor')


with open('model.bin', 'rb') as f_out:
    dv, final_model = pickle.load(f_out)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    X_input = dv.transform([data])
    
    dinput = xgb.DMatrix(X_input)
    
    prediction = final_model.predict(dinput)
    price = np.expm1(prediction)  
    return jsonify({'prediction': float(price[0])})

if __name__ == '__main__':
    app.run(debug=True)