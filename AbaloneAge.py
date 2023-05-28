import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import SGD
from Layer import Layer
from NN import NN
import time


# Load the Abalone dataset using pandas
abalone_data = pd.read_csv('./data/abalone.csv')

# Encode the "Sex" feature using one-hot encoding
abalone_data = pd.get_dummies(abalone_data, columns=['Sex'])

# Split the data into features and target
X = abalone_data.drop('Rings', axis=1)
y = abalone_data['Rings']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Remove outliers and missing values from X_train and y_train
X_train = X_train[X_train.Height < 0.4]
y_train = y_train.drop([1417, 2051])
X_train = X_train[X_train.Height != 0.00]
y_train = y_train.drop([1257, 3996])

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to NumPy arrays
X_train_scaled = np.array(X_train_scaled)
y_train = np.array(y_train)
X_test_scaled = np.array(X_test_scaled)
y_test = np.array(y_test)

# Reshape the input data for the custom neural network
X_train_reshaped = X_train_scaled.reshape(-1, 1, 10)
X_test_reshaped = X_test_scaled.reshape(-1, 1, 10)

# Create the Keras model
keras_model = Sequential()
keras_model.add(Dense(100, activation='relu', input_shape=(10,)))
keras_model.add(Dense(50, activation='relu'))
keras_model.add(Dense(1, activation='linear'))

# Set the learning rate for SGD optimizer
sgd_optimizer = SGD(learning_rate=0.001)
keras_model.compile(optimizer=sgd_optimizer, loss='mean_squared_error')

start_time_keras = time.time()
# Train the Keras model
keras_model.fit(X_train_scaled, y_train, epochs=100, batch_size=1, verbose=1)
end_time_keras = time.time()
runtime_keras = end_time_keras - start_time_keras

# Create the custom neural network
custom_net = NN()
custom_net.addLayer(Layer(10, 100, 'relu'))
custom_net.addLayer(Layer(100, 50, 'relu'))
custom_net.addLayer(Layer(50, 1, 'None'))

start_time_custom = time.time()
# Train the custom neural network
custom_net.Train(X_train_reshaped, y_train, epochs=100, learning_rate=0.001, Print=1)
end_time_custom = time.time()
runtime_custom = end_time_custom - start_time_custom

# Create the KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)

start_time_knn = time.time()
# Train the KNN model
knn_model.fit(X_train_scaled, y_train)
end_time_knn = time.time()
runtime_knn = end_time_knn - start_time_knn

# Predict using the keras model
keras_predictions = keras_model.predict(X_test_scaled)
keras_predictions = keras_predictions.flatten()
keras_rounded_predictions = np.round(keras_predictions).astype(int)
keras_mse = np.mean(np.square(y_test - keras_rounded_predictions))
print("Keras Model Mean Squared Error (MSE):", keras_mse)

# Predict using the custom neural network
custom_predictions = custom_net.predict(X_test_reshaped)
custom_predictions = np.array(custom_predictions).flatten()
custom_rounded_predictions = np.round(custom_predictions).astype(int)
print(custom_rounded_predictions.shape)
y_test_reshaped = y_test.reshape(custom_rounded_predictions.shape)
custom_mse = np.mean(np.square(y_test_reshaped - custom_rounded_predictions))
print("Custom Neural Network Mean Squared Error (MSE):", custom_mse)

# Predict using the KNN model
knn_predictions = knn_model.predict(X_test_scaled)
knn_predictions = np.round(knn_predictions).astype(int)
knn_mse = np.mean(np.square(y_test - knn_predictions))
print("KNN Model Mean Squared Error (MSE):", knn_mse)

# Create a bar chart comparing the MSE of the three models
models = ['Keras', 'Custom Neural Network', 'KNN']
mse_values = [keras_mse, custom_mse, knn_mse]

# Add runtimes to the list
runtimes = [runtime_keras, runtime_custom, runtime_knn]

plt.figure(figsize=(8, 6))
plt.bar(models, mse_values)
plt.xlabel('Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Comparison of MSE: Keras vs. Custom Neural Network vs. KNN')

# Add exact values and runtimes on top of each bar
for i, (v, r) in enumerate(zip(mse_values, runtimes)):
    plt.text(i, v, f"MSE: {round(v, 4)}\nRuntime: {round(r, 2)}s", ha='center', va='bottom')

plt.savefig('./results/mse_runtime_comparison_abalone.png')
plt.show()
