# A-Deep-LSTM-Network-for-Apple-Stock-s-Price-Forecasting
LSMT model to try to predict Apple’s stock price for a specific year using previous year’s data from NASDQ’s website
Stock price instabilities have a significant impact on many financial activities of the world. The development of a reliable prediction model could offer insights in stock price fluctuations, behavior and dynamics and ultimately could provide the opportunity of gaining significant profits. This paper explores the application of the LSTM model. This model exploits the ability of convolutional layers for extracting useful knowledge and learning the internal representation of time-series data as well as the effectiveness of (LSTM) layers for identifying short-term and long-term dependencies.
![image](https://user-images.githubusercontent.com/71279457/214450447-70766710-b4bc-421e-af01-40973026f19c.png)
(Orange: actual, Green: predicted)

Values are normalized in range (0,1).
- Datasets are split into train and test sets, 20% test data, 80% training data.
- Keras- TensorFlow is used for implementation.
- LSTM network consists of 25 hidden neurons, and 1 output layer (1 dense layer).
- LSTM network features input: 1 layer, output: 1 layer , hidden: 25 neurons, optimizer:adam, dropout:0.1, timestep:240, batchsize:240, epochs:1000 (features can be further optimized).
