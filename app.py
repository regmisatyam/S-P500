import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Loading and preprocessing data:
df = pd.read_csv('sp500_data.csv', skiprows=3, header=None)
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Parse 'Date' and set it as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Using only the 'Close' column for analysis
data = df[['Close']]

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data.loc[:, 'Close'] = scaler.fit_transform(data[['Close']])

# Prepare data for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

sequence_length = 60
X, y = create_sequences(data['Close'].values, sequence_length)

# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Create LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Visualize results
plt.figure(figsize=(16,8))
plt.plot(df.index[sequence_length:], y_train[0], label='Actual')
plt.plot(df.index[sequence_length:train_size], train_predict, label='Train Predict')
plt.plot(df.index[train_size+sequence_length:], test_predict, label='Test Predict')
plt.legend()
plt.title('S&P 500 Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# Calculate RMSE
train_rmse = np.sqrt(np.mean((train_predict - y_train[0])**2))
test_rmse = np.sqrt(np.mean((test_predict - y_test[0])**2))
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')
