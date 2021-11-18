# Coinbase Prediction Project

## Project Definition
In the last few years the importance of crypto currency has increased rapidly. Especially the last past month we saw Bitcoin, Ethereum and other currencies reaching their all time highs. More and more people are entering the market and the number of coins and platforms for trading these currencies are growing every day. Due to a huge personal interest in this field, I decided to dedicate my Capstone Project to this topic. This current project uses crypto currency data and builds a ML model, that predicts the closing price for a particular time period via webapp.  

### Data 
As a data source I use the API of Coinbase (see Acknowledgement down below), which delivers the prices of several crypto coins on given time and granularity.   

### Problem Statement
The aim of this project was to build a webapp that takes several input parameters and delivers data from the Coinbase API by using hourly granularity, stores the data in a Sqlite Database as well as a csv.-file. After getting the data from the data source a prediction model is trained and saved as a pickle file. This model is then used to predict the closing values for certain datetime values. The input parameters for pulling the data and training the model are the following:
- coin symbol or list of coin symbols (e.g. BTC or BTC;ETH for Bitcoin and Ethereum)
- currency (the currency in which the coin price should be pulled)
- start date for model training (date, e.g. 2017-01-01, to define the minimum day the dataset should contain)
- end date for model training (date, e.g. 2021-10-15, to define the maximum date for the model training process)

![grafik](https://user-images.githubusercontent.com/91085353/142240071-866aee48-873f-4941-a3fa-c1b209e5d6ba.png)

The API request gets the data from the defined start date until the current date (today). The "end date for model training" input parameter is used to limit the dataset used for training and testing. The reason of doing so is that we want to be able to predict the value for dates that are in the "future" referred to the dates in the training data.

The input data for the model trainig is the opening price, the high and the low of the particular time period (hour, because we have hourly granularity here).

After successful model training the webapp should be able to take further input values for the prediction:
- coin symbol or list of coin symbols (e.g. BTC or BTC;ETH for Bitcoin and Ethereum, the symbols should be part of the data processing step)
- datetime or list of datetimes (e.g. 2021-11-01 14;2021-11-01 18 for prediction the closing values for 2 and 6 pm on 01.Nov.2021)

![grafik](https://user-images.githubusercontent.com/91085353/142240160-5afe6a04-13b8-4f05-81cd-ab16e3b51bdd.png)

The predicted values should be displayed in the webapp.

### Metrics
The main metric used to evaluate the prediction models is the **mean deviation from the actual value**. This choice is reasonable, because all coins have different prices varying from less than a cent to over 60.000 USD. Using the percentage deviation from the actual value is independent of the coin price and allows to compare the metric among the different crypto coins. 

To compare the results using different ML techniques the **mean squared error**, **root mean squared error** and **mean absolute error** can be considered as well. The mean squared error penalizes larger errors, which is reasonable in our use case because a bigger difference in the coin price at the end of a time period can lead to worse consequences if one would use a prediction model for a investment strategy. Also we want to evaluate the root mean squared error as this metric also penalizes large errors but has the same unit as our dependent variable. 

Additionally we want to take a look at the mean absolute error because this metric gives us the actual average deviaton, as we want to know how many currency units (e.g. USD) we are deviating from the real closing price on average. This information is especially useful while planning investments using the prediction model and evaluating the actual risk. 

## Analysis
In this particular project the data analysis conducted on the input data was executed in a separate Jupyter Notebook ('CoinPricePrediction.ipynb').

### Data Exploration
The dataset used in the prediction model is being fetched every time a user uses the webapp and runs the data query and model training part with its own input parameters. Independent of the user input these following features are pulled by the API request:
- unix: can be interpreted as datetime 
- low: the lowest price in this particular time period
- high: the highest price in this particular time period
- open: the opening price in this particular time period
- close: the closing price in this particular time period
- volume: the traded volume in this particular time period 

![grafik](https://user-images.githubusercontent.com/91085353/142239439-19889a16-3696-4184-ab92-1147efbbb659.png)


The column to be predicted by the model is 'close'. The columns 'unix', 'open' and 'volume' are used as the features for the training model.
The exploratory part of this section is stored in the Jupyter Notebook 'CoinPricePrediction.ipynb'.

For data exploration which was made in the Jupyter Notebook I used a dataset that I pulled for the ETH coin from 2017-01-01 to 2021-10-15 on hourly granularity. 

This dataset contains 41129 data points. The minimum closing price in the dataset is 8.1$, while the maximum is about 600 times higher with a value of 4828,79$ and the mean of the closing price in this dataset is 643,14$.  

![grafik](https://user-images.githubusercontent.com/91085353/142402313-73854167-e92f-49af-84a7-4992e2a20692.png)

Also there are no missing values in this dataset due to the fact that the API delivers complete and consistent datapoints.

### Data Visualization
Data Visualizations can also be found in the Jupyter Notebook 'CoinPricePrediction.ipynb'.

## Methodology

### Data Preprocessing
A complex data preprocessing is not neccessary in this project. The data pulled from the Coinbase API is clean, has no missing values and can be used as is. Only the unix column that contains the datetime information is converted into the datetime type for further visualizations and data wrangling. However, the original unix column is also used in the prediction model, due to the fact that we want the datetime to be part of the model features and we need it as a numerical value.

### Implementation and Refinement
First of all, I implemented functions that can be used to access the Coinbase API and pull data on given coin, currency, start date and end date. Having the interface set up, I built the base structure of the webapp. I included some input fields and implemented a function that is executed after clicking a particular button, which pulls data and stores the data in a sqlite database. I wrapped all the data processing code into a Python script, so that it can be called by other scripts like the run.py that is running the webapp.

Moreover I added a visualization of the crypto currency price, that is displayed on the main page after the data processing and model training is done.

![grafik](https://user-images.githubusercontent.com/91085353/142239589-737ace8b-00c7-4816-8e01-80cf48d30e65.png)


For the modeling part I implemented another script, that is also called by the run.py app script. The modeling scripts reads in the requested data from the database and builds a model by using the input parameters for the time period given by the user. 

As a model I tried out a pipeline consisting of a StandardScaler combined with Linear Regression, a MLP Regressor and a Decision Tree Regressor to predict the closing prices of a given time hour of a particular day. 

Using the exemplary data set for the ETH coin from 2017-01-01 to 2021-10-15 the model using Linear Regression reached following metrics:
- mean squared error: 161.3668
- root mean squared error: 12.7030
- mean absolute error: 5.2478
- mean deviation from actual value: 0.78 %

After that first attempt I tried out the MLP Regressor and the Decision Tree Regressor that gave me following results compared to the Linear Regression approach:

|                                | Linear Regression| MLP Regressor | Decision Tree Regressor |
|------------------------------- |:----------------:|:-------------:|:-----------------------:|
|mean squared error              | 161.3668         | 169.828       | 344.5886                |
|root mean squared error         | 12.7030          | 13.0318       | 18.5630                 |
|mean absolute error             | 5.2478           | 5.7041        | 7.4579                  |
|mean deviation from actual value| 0.78 %           | 1.49%         | 1.05%                   |


As an attempt to improve the model I chose several parameters for the MLP Regressor and the Decision Tree Regressor and used GridSearch to find the best of these parameters.
For the MLP Regressor I varied the values of the alpha value, the maximum iterations and the type of learning rate and chose following parameteres:
- alpha : 0.0001, 0.0005
- max_iter: 200, 500
- learning_rate : 'constant', 'invscaling', 'adaptive' 

A comparison of the evaluation metrics with tuned vs. without tuned parameters by GridSearch is displayed in the table below. The best parameters that could be found by GridSearch are 0.0001 for alpha, 200 for max_iter and a invscaling learning rate.

|         MLP Regressor          | No parameter tuning | With parameter tuning | 
|--------------------------------|:-------------------:|:---------------------:|
|mean squared error              | 169.828             | 166.112               |
|root mean squared error         | 13.0318             | 12.8884               |
|mean absolute error             | 5.7041              | 5.6857                |
|mean deviation from actual value| 1.49%               | 1.23%                 |


We can see the metrics slightly improve using the tuned parameters, but compared to our first attempt using Linear Regression we did not get better result. 

So I started a third attempt using the Decision Tree Regressor while training the model. Here I also varied some of the parameters via GridSearch to find improvement. Following parameters I tried out:
- min_samples_leaf : 1, 2, 5 
- min_samples_split : 2, 5

The GridSearch revealed 5 to be the best value for the min_samples_leaf parameters and 2 for the min_samples_split parameter. The results of the model using the tuned parameters for the Decision Tree Regressor vs. no parameter tuning can be seen in the following table:

|    Decision Tree Regressor     | No parameter tuning | With parameter tuning | 
|--------------------------------|:-------------------:|:---------------------:|
|mean squared error              | 344.5886            | 234.7715              |
|root mean squared error         | 18.5630             | 15.3222               |
|mean absolute error             | 7.4579              | 6.3999                |
|mean deviation from actual value| 1.05%               | 0.89%                 |


The GridSearch also improved the result of the model using the Decision Tree Regressor regarding all metrics. Nevertheless it did not deliver better results compared to the Linear Regression approach.

As we can see in the evaluations above the pipeline including the Linear Regresson delivers better results regarding to the metrics we defined compared to the MLP Regressor and the Decision Tree Regressor. Therefore, implementing the prediction model for the webapp I decided to use Linear Regression in the pipeline.


## Results

After implementing all scripts we have a functioning webapp that is able to process input parameters and predict the closing price for a given timestamp and a given coin.
![grafik](https://user-images.githubusercontent.com/91085353/142241370-5a4be189-d1aa-4068-abeb-6e4764808188.png)

### Model Evaluation and Validation
While building the model, I was experimenting with different Regressors and also used GridSearch to find better parameters for the MLP Regressor and the Decision Tree Regressor. However, none of these approaches reached better metric than using simple Linear Regression in this case. Predicting the ETH Coin Linear Regression showed a mean deviation from acutual value of .78%, while the MLP Regressor reached 1.23% after using GridSearch to find better parameters. Experimenting with the Decision Tree Regressor, I saw that we can reach a mean deviation from actual value of .89%, which comes close to the Linear Regression. But comparing the mean squared error, root mean squared error and the mean absolute error, we can see that these values are higher which is an indicator for a less safe forecast.

![grafik](https://user-images.githubusercontent.com/91085353/142239908-09eb968c-16b9-49f2-8798-18c7b8678202.png)


### Justification
In the model evaluation part above we saw that in this particular case Linear Regression turned out to be the technique that delivered the best metics. The model uses very few features one of which correlated with the prediction column in a very strong way. Testing the same with another coin (Bitcoin in this case) showed the same results. Again Linear Regression reached the best metrics (for details metric values, see the 'CoinPricePrediction.ipynb' Jupyter Notebook).

## Conclusion

### Reflection
I enjoyed a lot doing this project because it combined so many interesting techniques like getting data from the web, storing it in a database, building a Machine Learning Model out of it and combine all of this as processes called by a webapp. The most interesting part in my optinion was finding the best prediction model to use in the webapp, but also building the webapp itself. Having an interactive interface bringing together all the python scpripts that process the data and build the model is a pleasing result doing a project like this.

### Improvement
Another approach of predicting the coin prices would be adding more features from different data sources to the model. One could extend the data pulling by e.g. using the Twitter API and counting the number of tweets related to crypto or using NLP techniques to process tweets from people influencing the crypto market like Elon Musk. Another interesting aspect could be analyzing the wheter one crypto currency has impact on the behavior of another coin price.

## Technical Information

### Installation
The project is based on Python 3.0 (Anaconda distribution. The following libraries are used:
1. numpy
2. pandas
3. sqlalchemy
4. sqlite3
5. nltk
6. sklearn
7. pickle
8. requests
9. json
10. datetime
11. math
12. matplotlib
13. seaborn


### Data 
The data used in this project is is from the Coinbase API. The data can be pulled from the following URL:
- https://api.pro.coinbase.com/

## Files in Repository
The folder "app" holds python scripts used inside the app (process_data.py and train_regressor.py) and the html files used for the flask webapp. The run.py file starts the app.
The python script train_regressor.py extracts data from the created database, builds a ML pipeline, trains a model and stores the model in the {coin}_regressor.pkl file in the "models" folder.

The folder "data" contains the sqlite database named CoinPricePrediction.db, the python file process_data.py that requests the data from the API, processes and loads the data into a SQL database and stores it additionally into a csv file. 

In the folder "notebook_items" contains an exemlary csv datafile for the ETH coin to be used in the Jupyter Notebook 'CoinPricePrediction.ipynb' as well as the models created during the notebook execution.

The Jupyter Notebook 'CoinPricePrediction.ipynb' itself is stored in the main folder together with the README.md file.

### Instructions:

1. Run the following command in the app's directory to run your web app.
    `python run.py`

2. Go to http://0.0.0.0:3001/ (instead of 0.0.0.0 I have to use my local IP)

3. Choose your input parameters (e.g. 'BTC', 'USD', '2017-01-01', '2021-10-15') and run the data processing and model training. After this step is finished you will see a plot with your chosen stock price on the main page.

4. Choose your prediction parameters (e.g. 'BTC', '2021-11-01') and run the prediction part. After the prediction is finished you will see the predicted values in a table at the bottom of the page


## Acknowledgements
1. [Coinbase](https://www.coinbase.com/) for providing an API that can be used for this project
2. [Udacity](https://www.udacity.com/) for teaching so much about data science and providing the chance to build a project of my own interest 

