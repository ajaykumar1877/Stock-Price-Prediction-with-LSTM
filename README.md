Stock Price Prediction using LSTM
This project uses a Long Short-Term Memory (LSTM) neural network to predict the closing price of a stock. The model is built using Keras/TensorFlow, and it fetches historical stock data from Yahoo Finance.

Description
1.The Python script stock_predictor.py performs the following steps:
2,Fetches Data: Downloads historical stock data for a specified ticker (e.g., AAPL for Apple Inc.) from Yahoo Finance.
3.Preprocesses Data: Cleans and scales the 'Close' price data to prepare it for the neural network.
4.Builds LSTM Model: Constructs a sequential LSTM model designed to learn from time-series data.
5.Trains the Model: Trains the model on 80% of the historical data.
6.Makes Predictions: Uses the trained model to predict stock prices on the remaining 20% of the data (the test set).
7.Visualizes Results: Plots the training data, actual stock prices, and the model's predictions on a single chart for comparison.
