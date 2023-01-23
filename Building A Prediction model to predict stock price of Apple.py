#!/usr/bin/env python
# coding: utf-8

# # Building A Prediction model to predict stock price of Apple.


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10

import tensorflow as tsrf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

from sklearn.preprocessing import MinMaxScaler


# Load the data
df = pd.read_csv('HistoricalData_1641407819284.csv')

df.head()


# # Data Manipulation
# Keep only the necessary columns

df = df[['Date', 'Close/Last']]
df.head()

# Convert the closing price to a float and the date to a datetime
df = df.replace({'\$':''}, regex = True)
df = df.astype({"Close/Last": float})
df["Date"] = pd.to_datetime(df.Date, format="%m/%d/%Y")
df.dtypes

# Set the date as the index
df.index = df['Date']
df

# # Data Visualization

plt.plot(df["Close/Last"],label='Close Price history')


# # Data Preparation

# Sort the data by date
df = df.sort_index(ascending=True,axis=0)

# Create a new dataframe with only the date and closing price columns
data = pd.DataFrame(index=range(0,len(df)-1),columns=['Date','Close/Last'])
for i in range(0,len(data)):
    data['Date'][i]=df['Date'][i]
    data['Close/Last'][i]=df['Close/Last'][i]
data.head()


# ## Min-Max Scaler

# Split the data into training and validation sets
train_data = data[:200]
valid_data = data[200:]

# Normalize the data using a MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
final_data = scaler.fit_transform(df[['Close/Last']])

# Create the training and validation sets
x_train_data, y_train_data = [], []
for i in range(60, len(train_data)):
    x_train_data.append(final_data[i-60:i,0])
    y_train_data.append(final_data[i,0])
x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compile and fit the LSTM model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train_data, y_train_data, epochs=100, batch_size=64)


# # Train and Test Data

#Get the predicted closing prices
inputs = data['Close/Last'][200:].to_numpy()
inputs = np.reshape(inputs,(-1, 1))
inputs = scaler.transform(inputs)
X_test = inputs.reshape(inputs.shape[0],1)


# # Predicted Function

closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)


# # Prediction Result

# Plot the results
valid_data['Predictions'] = closing_price
plt.plot(train_data['Close/Last'])
plt.plot(valid_data[['Close/Last','Predictions']])
plt.legend(['Train', 'Val', 'Prediction'], loc='upper left')
plt.show()


from sklearn.metrics import mean_absolute_error, mean_squared_error

#Calculate the MAE and MSE
mae = mean_absolute_error(valid_data['Close/Last'], valid_data['Predictions'])
mse = mean_squared_error(valid_data['Close/Last'], valid_data['Predictions'])

print("MAE: ", mae)
print("MSE: ", mse)


# # EXTRA


print("Shape of x_train_data: ", np.shape(x_train_data))
print("Shape of y_train_data: ", np.shape(y_train_data))

#Reshape the data for input into the LSTM
x_train_data = np.array(x_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0],x_train_data.shape[1],1))

#Compile and fit the LSTM
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(x_train_data, y_train_data, epochs=100, batch_size=64)

lstm_model.summary()


# In[ ]:




