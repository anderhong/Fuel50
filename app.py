from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

mul_reg = open("churn_regression_model.pkl", "rb")
ml_model = joblib.load(mul_reg)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            Cities_Auckland = float(request.form['Cities_Auckland'])
            Cities_Wellington = float(request.form['Cities_Wellington'])
            Cities_Christchurch = float(request.form['Cities_Christchurch'])
            Budget = float(request.form['Budget'])
            Profit = float(request.form['Profit'])
            Spending = float(request.form['Spending'])
            pred_args = [Budget,Profit,Spending,Cities_Auckland,Cities_Wellington,Cities_Christchurch]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            # mul_reg = open("multiple_regression_model.pkl", "rb")
            # ml_model = joblib.load(mul_reg)
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction = model_prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0')