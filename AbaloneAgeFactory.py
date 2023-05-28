import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers.legacy import SGD
from NNFactory import NNFactory
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
# Set custom LR for SGD optimizer to match the one from Custom Implementation
sgd_optimizer = SGD(learning_rate=0.0019724252452932925)

# Create the Keras model
# Parameters are set to match the best params for custom implementation
def create_keras_model():
    keras_model = Sequential()
    keras_model.add(Dense(66, activation='sigmoid', input_shape=(10,)))
    keras_model.add(Dense(41, activation='sigmoid'))
    keras_model.add(Dense(1, activation='linear'))
    keras_model.compile(optimizer=sgd_optimizer, loss='mean_squared_error')
    return keras_model


keras_regressor = KerasRegressor(build_fn=create_keras_model, epochs=30, batch_size=1, verbose=1)


start_time_keras = time.time()
# Train the KerasRegressor model
keras_regressor.fit(X_train_scaled, y_train)
end_time_keras = time.time()
runtime_keras = end_time_keras - start_time_keras

# Calculate cross-validation MSE for the KerasRegressor model
keras_cv_mse = -np.mean(cross_val_score(keras_regressor, X_train_scaled, y_train, scoring='neg_mean_squared_error', cv=5))

# Predict using the keras_regressor model
keras_predictions = keras_regressor.predict(X_test_scaled)

# Holdout evaluation
keras_mse = np.mean(np.square(y_test - keras_predictions))
print("Keras Model Cross-Validation Mean Squared Error (MSE):", keras_cv_mse)


# Create the custom neural network
model_custom = NNFactory.createNNByRandomizedSearch(
    X=X_train_reshaped,
    y=y_train,
    range_learning_rate=[0.001, 0.01],
    range_layers=[3, 3],
    range_epochs=[30, 30],
    layer_size_ranges=[[50, 100], [10, 50], [1, 1]],
    act_fn_deep=['sigmoid', 'relu'],
    act_fn_output=['None'],
    cv=KFold(n_splits=5),
    n_iter=10,
    print_interval=10,
    verbose=1
)

start_time_custom = time.time()
# Train the custom neural network -  holdout (mainly for runtime measuring)
model_custom.Train(X_train_reshaped, y_train, epochs=30, learning_rate=0.001, Print=1)
end_time_custom = time.time()
runtime_custom = end_time_custom - start_time_custom

# Predict using the custom neural network
custom_predictions = model_custom.predict(X_test_reshaped)
custom_predictions = np.array(custom_predictions).flatten()
custom_rounded_predictions = np.round(custom_predictions).astype(int)
print(custom_rounded_predictions.shape)
y_test_reshaped = y_test.reshape(custom_rounded_predictions.shape)
custom_cv_mse = np.mean(np.square(y_test_reshaped - custom_rounded_predictions))
print("Custom Neural Network Cross-Validation Mean Squared Error (MSE):", custom_cv_mse)

# Create the KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)

start_time_knn = time.time()
# Train the KNN model
knn_model.fit(X_train_scaled, y_train)
end_time_knn = time.time()
runtime_knn = end_time_knn - start_time_knn

# Calculate cross-validation MSE for the KNN model
knn_cv_mse = -np.mean(cross_val_score(knn_model, X_train_scaled, y_train, scoring='neg_mean_squared_error', cv=5))


# Predict using the KNN model
knn_predictions = knn_model.predict(X_test_scaled)
knn_predictions = np.round(knn_predictions).astype(int)
# Holdout evaluation
knn_mse = np.mean(np.square(y_test - knn_predictions))
print("KNN Model Cross-Validation Mean Squared Error (MSE):", knn_cv_mse)

# Create a bar chart comparing the MSE of the three models
models = ['Custom Neural Network', 'Keras', 'KNN']
mse_cv_values = [custom_cv_mse, keras_cv_mse, knn_cv_mse]

# Add runtimes to the list
runtimes = [runtime_custom, runtime_keras, runtime_knn]

plt.figure(figsize=(8, 6))
plt.bar(models, mse_cv_values)
plt.xlabel('Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Comparison of MSE: Custom Neural Network vs. Keras vs. KNN')
plt.ylim(0,6)
# Add exact values and runtimes on top of each bar
for i, (v, r) in enumerate(zip(mse_cv_values, runtimes)):
    plt.text(i, v, f"MSE: {round(v, 4)}\nRuntime: {round(r, 2)}s", ha='center', va='bottom')

plt.savefig('./results/crossval_mse_runtime_comparison_abalone.png')
plt.show()
