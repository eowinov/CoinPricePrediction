import sys
import pandas as pd
import requests
import json
import datetime
from datetime import timedelta
from datetime import datetime
import math
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sqlalchemy import create_engine
import pickle
import sqlite3


def load_data(database_filepath, coin, end_date_train):
    '''
    INPUT:
    database_filepath - the filepath of the database that contains needed data
    coin - the coin name for which the data is fetched from data base
    end_date_train - maximal date in the training and test data
    
    OUTPUT:
    X - Dataframe that contains features for predicition model
    Y - Column containg closing price for an hour, that will be predicted by model
    
    Reads data from table of SQL Database for the particular coin, limits the date range of 
    train and test data to end_date_train and creates the feature and label DataFrames.
    '''
    # load data from database
    engine = create_engine('sqlite:///../{}'.format(database_filepath))

    conn = sqlite3.connect('../{}'.format(database_filepath))

    df = pd.read_sql_table('{}'.format(coin), engine)


    df = df[df['date'] <= pd.to_datetime(end_date_train)]


    # create feature and label dataframes
    X = df[['unix','open','volume']]
    Y = df['close']
    
    return X,Y
    


def build_model():
    '''
    OUTPUT:
    pipeline - a pipeline containing the StandardScaler and LinearRegression
    
    Creates a pipeline that contains the StandardScaler and LinearRegression.
    '''

    pipeline = Pipeline([
        ('stdscale', StandardScaler()),
        ('clf', LinearRegression())
    ])

   
    return pipeline

def print_score_values(y_test, y_pred):
    '''
    INPUT:
    y_test - a dataframe containing the labels of the test data
    y_pred - a dataframe containing the predicted labels
        
    Calculates the mean deviation from acutal value and prints the results containing
    mean squared error, root mean squared error, mean absolute error and the mean deviation
    from actual value.
    '''
    diff = abs(y_test - y_pred)
    deviation = diff / y_test

    mean_deviation = deviation.sum().sum() / y_test.shape[0]

    print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error: ', math.sqrt(mean_squared_error(y_test, y_pred)))
    print('Mean Absolute Error: ',mean_absolute_error(y_test, y_pred))
    print('Mean Deviation from actual value: {}%'.format(round(mean_deviation*100,2)))


def evaluate_model(model, X_test, Y_test):
    '''
    INPUT:
    model - trained model that has to be evaluated
    X_test - dataframe containing the test data
    Y_test - a dataframe containing the labels of the test data


    Predicts the labels using the test dataframe and creating a new Dataframe out 
    of the predicted labels, calls the function that prints the score values of model
    '''
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.Series(Y_pred)
    
    Y_pred_df.index = Y_test.index

    Y_pred_df[Y_pred_df < 0] = 0

    Y_pred_df = round(Y_pred_df, 2)

    print_score_values(Y_test, Y_pred_df)


def save_model(model, model_filepath):
    '''
    INPUT:
    model - trained model that has to be saved
    model_filepath - filepath to save the model at
        
    Saves the model as a pickle file under the given filepath
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def train(database_filepath, model_filepath, coin, end_date_train):
    '''
    INPUT:
    database_filepath - the filepath of the database that contains needed data
    model_filepath - filepath to save the model at
    coin - the coin name for which the data is fetched from data base and a model is build
    end_date_train - maximal date in the training and test data
        
    Loads the data for the particular coin from the database, builds, fits and evaluated the model
    and saves the model as a pickle file under the given filepath.
    '''

    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y = load_data(database_filepath, coin, end_date_train)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print('Building model for {}...'.format(coin))
    model = build_model()
    
    print('Training model for {}...'.format(coin))
    model.fit(X_train, Y_train)

    #best_model = model.best_estimator_
    
    print('Evaluating model for {}...'.format(coin))
    evaluate_model(model, X_test, Y_test)

    print('Saving model for {}...\n    MODEL: {}'.format(coin, model_filepath))
    save_model(model, model_filepath)

    print('Trained model for {} saved!'.format(coin))


def main():
    if len(sys.argv) == 5:
        train(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print('Please provide the filepath of the coin price database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument and the coin symbol '\
              'as the third argument. \n\nExample: python '\
              'train_regressor.py ../data/CoinPricePrediction.db regressor.pkl BTC')


if __name__ == '__main__':
    main()