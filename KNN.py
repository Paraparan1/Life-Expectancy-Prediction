import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt


np.random.seed(32138)


# Load the dataset
data = pd.read_csv('Life_Expectancy_Cleaned_Data.csv')

# Perform one-hot encoding on 'economic_status' column
status_onehot = pd.get_dummies(data['economic_status'])

# Perform label encoding on 'country_name' column
label_encoder = LabelEncoder()
data['country_encoded'] = label_encoder.fit_transform(data['country_name'])

# Combine the one-hot encoded 'economic_status' column with the original data
data = pd.concat([data, status_onehot], axis=1)

# Split the dataset into input features (X) and target variable (y)
X_train= data.drop(['life_expectancy', 'economic_status', 'country_name'], axis=1)
y_train = data['life_expectancy']

# Normalize the input features using z-score normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Define the range of k values to test
k_values = list(range(1, 16))

# Define the number of folds for cross-validation
num_folds = 10

# Loop over the k values and compute the mean squared error (MSE) for each one
mse_values = []
accuracy_values = []
training_times = []


for k in k_values:

    knn = KNeighborsRegressor(n_neighbors=k, metric="manhattan")
    k_fold= KFold(n_splits=num_folds, shuffle=True, random_state=42)

    start_time = time.time()  # Record start time
    mse = -1 * cross_val_score(knn, X_train, y_train, cv=k_fold, scoring='neg_mean_squared_error')#Returns a negative mse value
    mse_values.append(mse.mean())
    end_time = time.time()  # Record end time

    #Calculate training time
    training_time = end_time - start_time  # Calculate training time

    #append training time
    training_times.append(training_time)

    # Calculate RMSE and accuracy
    rmse = np.sqrt(mse.mean())
    accuracy = 1 / rmse
    accuracy_values.append(accuracy)

    #Print results
    print('For k = {}: RMSE = {:.4f}, Accuracy = {:.4f}, Time = {:.4f}'.format(k, rmse, accuracy,training_time))


# Plot the MSE values against the k values
plt.figure(figsize=(10, 6))
plt.plot(k_values, mse_values, '-o')
plt.xlabel('k Value')
plt.ylabel('Mean Squared Error')
plt.title('KNN Model Performance: MSE vs. k')
plt.grid(True)
plt.show()

# Plot accuracy against k values
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_values, '-o', label='Accuracy')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.title('KNN Model Performance: Accuracy vs. k')
plt.legend()
plt.grid(True)
plt.show()


# Plot training time against k values
plt.plot(k_values, training_times, '-o', label='Training Time')
plt.xlabel('k Value')
plt.ylabel('Training Time (s)')
plt.title('KNN Model Performance: Training Time vs. k')
plt.legend()
plt.show()