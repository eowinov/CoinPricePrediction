import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter 
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine
import sqlite3

import process_data as p
import train_regressor as t

import datetime
import math


app = Flask(__name__)


# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # render web page 
    return render_template('master.html')


# web page that handles user input and triggers data processing and model building
@app.route('/go')
def go():
    # save user input in variables
    coin_symbols_train = request.args.get('coin_symbols_train', '')
    currency = request.args.get('currency', '') 
    start_date_train = request.args.get('start_date', '') 
    end_date_train = request.args.get('end_date', '') 
    
    # define current date for data request, preparing input for further processes
    date_today = datetime.date.today()
    date_today_str = str(date_today.year) + '-' + str(date_today.month) + '-' + str(date_today.day)
    end_date_train_split = end_date_train.split('-')
    end_date = datetime.date(year=int(end_date_train_split[0]), month=int(end_date_train_split[1]), day=int(end_date_train_split[2]))
    coins = coin_symbols_train.split(';')

    # loop through all coins, process the data and build the model
    for coin in coins:
        symbol = coin + '-' + currency
        p.process('sqlite:///../data/CoinPricePrediction.db', coin,currency,start_date_train,date_today_str)
        t.train('data/CoinPricePrediction.db', '../models/{}_regressor.pkl'.format(coin),coin, end_date)

    #load data for visuals
    engine = create_engine('sqlite:///../data/CoinPricePrediction.db')
    graphs = []
    for coin in coins:
        df = pd.read_sql_table(str(coin), engine)
        print('read table ', str(coin))
        graph = {
            'data': [
                Scatter(
                    x=df.index.values,
                    y=df['close']
                )
            ],
            'layout': {
                'title': str(coin),
                'yaxis': {
                    'title': "Price"
                },
                'xaxis': {
                    'title': "Time"
                }
            }
        }
        graphs.append(graph)
    print('created graphs')

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

    
# web page that handles user input and predicts coin price
@app.route('/go_predict')
def go_predict():
    coin_pred = request.args.get('coin_pred', '')
    dates_pred = request.args.get('dates_pred', '')
    
    coins = coin_pred.split(';')
    dates = list(dates_pred.split(';'))
    results = dict()

    engine = create_engine('sqlite:///../data/CoinPricePrediction.db')

    # loop trough coins, load model and predict values based on input parameters
    for coin in coins:
        coin_results = dict()
        data = pd.read_sql_table(str(coin), engine)
        model = joblib.load("../models/{}_regressor.pkl".format(coin))

        data = data[data['date'].isin(dates)]
        print(data.head())
        data = data[['unix','open','volume']]

        y_pred = model.predict(data)
        print(y_pred[0])

        for d in range(len(dates)):
            coin_results[dates[d]] = np.around(y_pred[d],2)

        print(coin)
        print(coin_results)
        results[coin] = coin_results

    print(results)

    # This will render the go_predict.html  
    return render_template(
        'go_predict.html',
        coin_pred = coin_pred,
        results = results 
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()