import sys
import pandas as pd
import numpy as np
import requests
import json
import datetime
from datetime import timedelta
import math
from sqlalchemy import create_engine


def fetch_daily_data(symbol, from_date, to_date):
    '''
    INPUT:
    symbol - coin name and currency in which the price is displayed, e.g. BTC-USD
    from_date - minimal date that should be fetched from data source
    to_date - maximal date that should be fetched from data source
    
    OUTPUT:
    data_concat - a dataframe that cointains the data fetched from the Coinbase API with date between
                  from_date and to_date
    
    Requests the Coinbase API with given date parameters and provides a dataframe that contains data for the 
    given coin, in the given time range on hourly granularity.
    '''

    pair_split = symbol.split('-')  
    
    from_date_str = str(from_date)
    to_date_str = str(to_date)
    
    granularity = 3600
    
    url = f'https://api.pro.coinbase.com/products/{symbol}/candles?start={from_date_str}&end={to_date_str}&granularity={granularity}'
    coinbase_data = requests.get(url)

    if coinbase_data.status_code == 200:  
        # load data into dataframe and create new date column out of unix column
        data = pd.DataFrame(json.loads(coinbase_data.text), columns=['unix', 'low', 'high', 'open', 'close', 'volume'])
        data['date'] = pd.to_datetime(data['unix'], unit='s')  

        if data is None:
            print("Did not return any data from Coinbase for this symbol")
            return None
        else:
            
            return data

    else:
        print("Did not receieve OK response from Coinbase API")
        return None

def fetch_data_time_period(symbol,from_date,to_date):
    '''
    INPUT:
    symbol - coin name and currency in which the price is displayed, e.g. BTC-USD
    from_date - minimal date that should be fetched from data source
    to_date - maximal date that should be fetched from data source
    
    OUTPUT:
    data_concat - a dataframe that cointains the data fetched from the Coinbase API with date between
                  from_date and to_date
    
    Calculates how many times the API request has to be sent (API handles max. 300 rows at a time), 
    splits the date range in appropriate part and calls the function that requests the data for each 
    particular time range. Concats the fetched data parts and provides a DataFrame containing the data 
    between from_date and to_date.
    '''
    
    data = []
    
    #calculate days between from_date and to_date
    split_from_date = from_date.split('-')
    split_to_date = to_date.split('-')
    
    # create date objects out of date strings
    from_date = datetime.date(int(split_from_date[0]),int(split_from_date[1]),int(split_from_date[2]))
    to_date = datetime.date(int(split_to_date[0]),int(split_to_date[1]),int(split_to_date[2]))
    
    
    time_between = to_date - from_date
    
    # calculate number of iterations for api request (300 datapoints per request allowed)
    n = math.ceil(time_between.days * 24 /300)

    
    for i in range(n):
        from_date_iter = from_date + timedelta(hours= i * 300)
        to_date_iter = from_date + timedelta(hours = (i+1) * 300 - 1)
            
        data_iter = fetch_daily_data(symbol, from_date_iter, to_date_iter)
        data_iter = data_iter.sort_values(by='date')
        data.append(data_iter)

    data_concat = pd.DataFrame()
    try:
        data_concat = pd.concat(data)
        data_concat = data_concat.reset_index(drop=True)
    except: 
        ValueError

    return data_concat   


def save_data(data, coin, currency, database_filename):
    '''
    INPUT:
    data - a dataframe that is saved in a database
    coin - the coin name for which the data is saved in data based
    currency - the currency in which the data for this coin is stored
    database_filename - the name of the file that contains database
    
    Creates a new engine and saves the dataframe to a SQL Database in a table for this particular coin and currency
    and additionally to a csv file.
    '''
    engine = create_engine(database_filename)
    data.to_sql(str(coin), engine, if_exists='replace', index=False)  
    data.to_csv(f'Coinbase_{coin}_{currency}.csv', index=False)


def process(database_filepath, coin, currency, start_date, end_date):
    '''
    INPUT:
    database_filepath - filepath to the database the data should be stored 
    coin - the coin name for which the data is fetched and saved in data base
    currency - the currency in which the data for this coin is fetched and stored
    start_date - minimal date that should be fetched from data source
    end_date - maximal date that should be fetched from data source
    
    Calls all other functions to fetch the data, and save it into a database and as
    a csv file.
    '''

    print('Loading data for {}...'.format(coin))

    symbol = coin + '-' + currency
    data = fetch_data_time_period(symbol,start_date,end_date)
    
    print('Saving data for {}...\n    DATABASE: {}'.format(coin, database_filepath))
    save_data(data, coin, currency, database_filepath)
    
    print('Data for {} saved to database!'.format(coin))
    

def main():
    if len(sys.argv) == 5:

        process(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()