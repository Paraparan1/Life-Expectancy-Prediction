import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.losses import MeanSquaredError
from sklearn.metrics import r2_score
import time

# Setting random seed to create reproducible results.
np.random.seed(32138)

# Load the dataset
data = pd.read_csv('Life_Expectancy_Cleaned_Data.csv')

# Perform one-hot encoding on 'economic_status' column
status_onehot = pd.get_dummies(data['economic_status'])

# Perform label encoding on 'country_name' column
label_encoder = LabelEncoder()
data['country_encoded'] = label_encoder.fit_transform(data['country_name'])

# Combine the one-hot encoded 'Status' column with the original data
data = pd.concat([data, status_onehot], axis=1)

# Split the dataset into input features (X) and target variable (y)
X = data.drop(["life_expectancy", "economic_status", "country_name"], axis=1)
y = data["life_expectancy"].values

# Normalize input data using minmax scaler:
# Each feature are transformed to the range [0, 1] by subtracting the minimum value and dividing by the range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Split into training, validation, and testing sets 70% for training 30% for testing and validation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Reshape input data for CNN
X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.values.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

#Build CNN model
model = Sequential()

# Add 2 1D Convolutional layer with 128 filters and a kernel size of 4, with ReLU activation function
model.add(Conv1D(filters=128, kernel_size=4, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Conv1D(filters=128, kernel_size=4, activation='relu'))

#Max pooling layer to reduce the spatial dimenssions for feature map.
model.add(MaxPooling1D(pool_size=2))

# Add a Flatten layer to flatten the output from previous layers into a 1D vector
model.add(Flatten())

# Add a Dense layer with 150 neurons and ReLU activation function
model.add(Dense(150, activation='relu'))

# Add a Dropout layer with dropout rate of 0.2 to prevent overfitting
model.add(Dropout(0.2))
model.add(Dense(1))

#Print model architecture
model.summary()

# Compile model
model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=['mse', 'mae', 'mape'])

# Record Start time
startime = time.time()


#Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Evaluate model
loss, mse, mae, mape = model.evaluate(X_test, y_test, verbose=0)
metrics = model.evaluate(X_test, y_test, verbose=0)

endtime = time.time() # Record endtime

training_time = endtime - startime

model.save('LifeExpectancyModel4.h5')

# Print evaluation results
print("Evaluation results:")
print("Training Time: {:.2f}s".format(training_time))
print("Loss: {:.4f}".format(loss))
print("MSE: {:.4f}".format(mse))
print("MAE: {:.4f}".format(mae))
print("MAPE: {:.4f}".format(mape))

#As MSE was used as the loss value we can plot the square root of mse to determine model performance.
rmse_values = np.sqrt(history.history['val_mse'])


# Plot RMSE against epochs
plt.plot(range(1, len(rmse_values) + 1), rmse_values)
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Validation RMSE vs. Epochs')
plt.show()

print('Final RMSE:', rmse_values[-1])

# Testing the predictions.
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

# The closer the r score is to one the better the fit.
print("R-squared (R2): {:.4f}".format(r2))


# Plot true values (y_test)
plt.plot(y_test, label='True Values')

# Plot predicted values (y_pred)
plt.plot(y_pred, label='Predicted Values')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('True Values vs. Predicted Values')
plt.legend()

# Show the plot
plt.show()


# Plot true values distribution
plt.hist(y_test, bins=20, alpha=0.5, label='True Values')

# Plot predicted values distribution
plt.hist(y_pred, bins=20, alpha=0.5, label='Predicted Values')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('True Values and Predicted Values Distribution')
plt.legend()
plt.show()

