#Importing Independancies 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import pandas_datareader as pdr
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

#!pip install yfinance
import yfinance as yf
from datetime import datetime

# #Geting data from Yahoo Finance
# # Set the date range
# # Define the ticker symbol and the date range
# ticker = 'NVDA'
# end_date = datetime.now()
# start_date = datetime(end_date.year - 2, end_date.month, end_date.day)

# # Fetch the historical stock prices
# nvda_data = yf.download(ticker, start=start_date, end=end_date)

# # Display the first few rows to check
# print(nvda_data.head())

# # Extract the 'Close' column and convert to a list
# closing_prices = nvda_data['Close'].tolist()

# # Display the first few closing prices to verify
# print(closing_prices[:5])


#Loading data that was provided
data = pd.read_csv('C:\\Development\\AtB interview assignment\\NVDA.csv')
print(data.head())

'''Data Cleaning and preprocessing'''

# Convert the date column to datetime format if it's not already
if 'Date' in data.columns and data['Date'].dtype == object:
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Convert all other columns to numeric, coercing errors to NaN
for column in data.columns:
    if column != 'Date':  # Skip the date column
        data[column] = pd.to_numeric(data[column], errors='coerce')

#handle missing values using mean
data = data.fillna(data.mean())
#remove duplicates
data = data.drop_duplicates()
#Finding outliers using IQR
Q1 = data['Close'].quantile(0.25)
Q3 = data['Close'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

#removing outliers
data = data[(data['Close'] >= lower_bound) & (data['Close'] <= upper_bound)]
# Display the cleaned data
print("Cleaned Data:")
print(data.head())


'''Feature Engineering'''

# Lag Features
for lag in range(1, 4):  # Create lags for 1, 2, 3 days
    data[f'lag_{lag}'] = data['Close'].shift(lag)

# Rolling Window Features
data['rolling_mean_7'] = data['Close'].rolling(window=7).mean()
data['rolling_std_7'] = data['Close'].rolling(window=7).std()

# Exponential Moving Average
data['ema_7'] = data['Close'].ewm(span=7, adjust=False).mean()

# Percentage Change
data['pct_change'] = data['Close'].pct_change()

# Implement RSI - simplifying assumptions for RSI
delta = data['Close'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
data['rsi'] = 100 - (100 / (1 + (up.rolling(14).mean() / down.rolling(14).mean())))


# Daily Return
data['daily_return'] = data['Close'].pct_change()

# Cumulative Return
data['cumulative_return'] = (1 + data['daily_return']).cumprod() - 1

# Display the enhanced dataset with new features
print(data.head(10))


#Train-Test Split
# Split the data into training and testing sets without shuffling
train, test = train_test_split(data, test_size=0.2, shuffle=False)

print(train.head())

# Assuming 'data' is your DataFrame and 'Close' is the column we're predicting
values = data['Close'].values.reshape(-1, 1)

#Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train[['Close']])
test_scaled = scaler.transform(test[['Close']])

# Function to create a dataset for LSTM
def create_dataset(data, look_back):
    X, Y = [], []
    for i in range(len(data) - look_back):
        a = data[i:(i + look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 30  # Number of previous time steps to consider for predicting the next time step
X, Y = create_dataset(train_scaled, look_back)
X_test, Y_test = create_dataset(test_scaled, look_back)

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
#building model
# model = Sequential()
# model.add(LSTM(50, return_sequences=True, input_shape=(1, look_back))) model.add(Dropout(0.2))
# model.add(LSTM(50, return_sequences=False))  # Ensure the last LSTM layer has return_sequences=False
# model.add(Dropout(0.2))
# model.add(Dense(1))

''' Building an improved model'''
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))  # First LSTM layer with 100 neurons
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))  # Second LSTM layer with 100 neurons
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))  # Third LSTM layer with 50 neurons
model.add(Dropout(0.2))
model.add(Dense(1))



model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
# Assume X and Y are already prepared with the correct shape
# X = x train
# y = y train
model.fit(X, Y, epochs=150, batch_size=1, verbose=2, validation_data=(X_test, Y_test))

#Evaluation

from tensorflow.keras.metrics import MeanSquaredError
# Assuming you have already compiled your model with MSE as a metric, and have trained it
# Now predict the stock prices using the model
predictions = model.predict(X_test)

# Calculate Mean Squared Error using keras' built-in functionality
mse_metric = MeanSquaredError()
mse_metric.update_state(Y_test, predictions)
mse = mse_metric.result().numpy()

# Calculate Root Mean Squared Error
rmse = np.sqrt(mse)

print(f"Mean Squared Error on Test Set: {mse:.4f}")
print(f"Root Mean Squared Error on Test Set: {rmse:.4f}")


import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
# Plot actual closing prices
plt.plot(test.index, scaler.inverse_transform(test[['Close']]), label='Actual')

# Plot predicted closing prices
plt.plot(test.index[look_back:], scaler.inverse_transform(predictions), label='Predicted')

plt.title('Actual vs Predicted Closing Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()



# Retrieve the last 'look_back' days from the test data for the prediction
last_batch = test_scaled[-look_back:]
last_batch = last_batch.reshape((1, 1, look_back))

# Make a prediction
predicted_scaled_price = model.predict(last_batch)

# Inverse transform to get the actual predicted stock price
predicted_price = scaler.inverse_transform(predicted_scaled_price)
print(f"Predicted NVDA closing price for the next day: {predicted_price[0][0]}")






