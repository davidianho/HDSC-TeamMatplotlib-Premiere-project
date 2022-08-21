import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from model import country_list_gen, prediction_model, selecting_country, prediction 


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [(x) for x in request.form.values()]
    country = features[0].lower()
    year = int(features[1])
    df = pd.read_csv('data.csv')
    lists, df = country_list_gen(df)
    if country in lists:
        df = selecting_country(df, country)
        model = prediction_model(df)
        result = prediction(model,year)
        return render_template('index.html', prediction_text=f"\n Result: {country.upper()} population in {year} will be {result:,d}")
    else:
        return render_template('index.html', prediction_text='invalid country name or year !')


if __name__ == "__main__":
    app.run(debug=True)