import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from Layer import Layer
from NN import NN

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
X_train_reshaped = X_train_scaled.reshape(-1, 1, 10)  # Reshape to have shape (num_samples, 1, num_features)

# Create the Keras model
keras_model = Sequential()
keras_model.add(Dense(100, activation='relu', input_shape=(10,)))
keras_model.add(Dense(50, activation='relu'))
keras_model.add(Dense(1, activation='linear'))
keras_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the Keras model
keras_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)

# Create the custom neural network
custom_net = NN()
custom_net.addLayer(Layer(10, 100, 'relu'))
custom_net.addLayer(Layer(100, 50, 'relu'))
custom_net.addLayer(Layer(50, 1, 'None'))

# Train the custom neural network
custom_net.Train(X_train_reshaped, y_train, epochs=100, learning_rate=0.001, Print=10)

# Preprocess the test data using the scaler fitted on the training data
X_test_scaled = scaler.transform(X_test)

# Predict using the keras model
keras_predictions = keras_model.predict(X_test_scaled)
keras_predictions = keras_predictions.flatten()
keras_rounded_predictions = np.round(keras_predictions).astype(int)
keras_mse = np.mean(np.square(y_test - keras_rounded_predictions))
keras_df_predictions = pd.DataFrame({'Actual': y_test, 'Prediction': keras_rounded_predictions})
print("Keras Model Predictions:")
print(keras_df_predictions)
print("Keras Model Mean Squared Error (MSE):", keras_mse)

# Predict using the custom neural network
custom_predictions = custom_net.predict(X_test_scaled)
custom_predictions = np.array(custom_predictions).flatten()
custom_rounded_predictions = np.round(custom_predictions).astype(int)
custom_mse = np.mean(np.square(y_test - custom_rounded_predictions))
custom_df_predictions = pd.DataFrame({'Actual': y_test, 'Prediction': custom_rounded_predictions})
print("Custom Neural Network Predictions:")
print(custom_df_predictions)
print("Custom Neural Network Mean Squared Error (MSE):", custom_mse)


# Create a bar chart comparing the MSE of the two models
models = ['Keras', 'Custom Neural Network']
mse_values = [keras_mse, custom_mse]

plt.bar(models, mse_values)
plt.xlabel('Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Comparison of MSE: Keras vs. Custom Neural Network')
plt.savefig('./results/mse_comparison_abalone.png')
plt.show()