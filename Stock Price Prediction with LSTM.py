# Import required libraries
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
# Updated Keras imports to work with modern TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Step 1: Load the stock data
# Choose the stock ticker symbol (e.g., "AAPL" for Apple)
stock = "AAPL"
data = yf.download(stock, start="2010-01-01", end="2023-01-01")

# Display the data
print("--- Stock Data Head ---")
print(data.head())
print("-----------------------\n")


# Step 2: Data Preprocessing
# We'll use only the 'Close' price for prediction
data = data[['Close']]

# Fill any missing values
data.fillna(method='ffill', inplace=True)

# Convert the data into a NumPy array
dataset = data.values

# Scale the data using MinMaxScaler (LSTM performs better with scaled data)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


# Step 3: Prepare the training data
# Use 80% of data for training
training_data_len = int(np.ceil(len(scaled_data) * 0.8))

# Create the training data
train_data = scaled_data[0:int(training_data_len), :]

# Split the data into x_train and y_train datasets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    # 60 previous values as features
    x_train.append(train_data[i-60:i, 0])
    # The next value as target
    y_train.append(train_data[i, 0])

# Convert x_train and y_train to NumPy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape x_train for LSTM model input (samples, timesteps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Step 4: Build the LSTM model
model = Sequential()

# Add LSTM layers
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))

# Add dense layers
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# Step 5: Train the model
print("--- Training the Model ---")
# For a real application, you'd want more epochs, but 1 is fine for a quick test.
model.fit(x_train, y_train, batch_size=1, epochs=1)
print("--- Training Complete ---\n")


# Step 6: Create the test dataset
# Create a new array containing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_len - 60:, :]

# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :] # Actual prices

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert x_test to a NumPy array
x_test = np.array(x_test)

# Reshape x_test for LSTM model input
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# Step 7: Get the model's predicted price values
print("--- Making Predictions ---")
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
print("--- Predictions Complete ---\n")


# Step 8: Visualize the results
# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid.loc[:, 'Predictions'] = predictions


# Visualize the predicted stock prices with the actual stock prices
plt.figure(figsize=(16,8))
plt.title('Stock Price Prediction Model for ' + stock)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Actual', 'Predicted'], loc='lower right')
plt.grid(True)
plt.show()

# Display the valid and predicted prices
print("--- Validation and Predicted Prices ---")
print(valid)
print("---------------------------------------")

