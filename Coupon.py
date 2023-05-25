import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Import the custom neural network implementation
from NN import NN
from Layer import Layer

# Load the In-vehicle Coupon Recommendation dataset
coupon_data = pd.read_csv('./data/in-vehicle-coupon-recommendation.csv')

print(coupon_data.columns)

# Split the data into features (X) and target (y) first
X = coupon_data.drop('Y', axis=1)
y = coupon_data['Y']
y = y.astype(int)

# Categorical columns that need one-hot encoding
categorical_columns = ['destination', 'passanger', 'weather', 'coupon', 'gender', 'age', 'maritalStatus', 'education', 'occupation', 'income', 'car', 'Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50']

# Select numerical columns for scaling
numerical_columns = ['temperature', 'expiration', 'time']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Replace missing values in training set with the most frequent value
X_train = X_train.fillna(X_train.mode().iloc[0])

# Replace missing values in test set with the most frequent value
X_test = X_test.fillna(X_train.mode().iloc[0])

# Convert categorical variables to numerical using one-hot encoding
X_train_encoded = pd.get_dummies(X_train, columns=categorical_columns)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns)

# Replace time values for better comparison
X_train_encoded = X_train_encoded.replace(['6PM', '7AM', '10AM', '2PM', '10PM'], [18, 7, 10, 14, 22])
X_test_encoded = X_test_encoded.replace(['6PM', '7AM', '10AM', '2PM', '10PM'], [18, 7, 10, 14, 22])

# Replace expiration time values with hours for better comparison
X_train_encoded = X_train_encoded.replace(['1d', '2h'], [24, 2])
X_test_encoded = X_test_encoded.replace(['1d', '2h'], [24, 2])

# Scale the numerical columns only
scaler = StandardScaler()
X_train_scaled = X_train_encoded.copy()
X_test_scaled = X_test_encoded.copy()
X_train_scaled[numerical_columns] = scaler.fit_transform(X_train_encoded[numerical_columns])
X_test_scaled[numerical_columns] = scaler.transform(X_test_encoded[numerical_columns])

X_train_scaled = X_train_scaled.astype(np.float32).values  # Convert to numpy array
y_train = y_train.astype(np.int32).values  # Convert to numpy array
X_test_scaled = X_test_scaled.astype(np.float32).values  # Convert to numpy array
y_test = y_test.astype(np.int32).values  # Convert to numpy array

# Reshape the input data
X_train_reshaped = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
X_test_reshaped = X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1])

# Print the number of features and instances
num_features = X_train_reshaped.shape[2]
num_instances = X_train_reshaped.shape[0]
print("Number of features:", num_features)
print("Number of instances:", num_instances)

# Create the KEras model
keras_model = Sequential()
keras_model.add(Dense(100, activation='relu', input_shape=(num_features,)))
keras_model.add(Dense(50, activation='relu'))
keras_model.add(Dense(1, activation='sigmoid'))

# Compile the Keras model
keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Keras model
keras_history = keras_model.fit(X_train_scaled, y_train, epochs=100, batch_size=64, verbose=1)

# Evaluate the Keras model on the test set
keras_loss, keras_accuracy = keras_model.evaluate(X_test_scaled, y_test)
print("Keras Model - Test Loss:", keras_loss)
print("Keras Model - Test Accuracy:", keras_accuracy)

# Create the custom neural network model
custom_model = NN()
custom_model.addLayer(Layer(num_features, 100, activation='relu'))
custom_model.addLayer(Layer(100, 50, activation='relu'))
custom_model.addLayer(Layer(50, 1, activation='sigmoid'))

# Train the custom neural network model
custom_model.Train(X_train_reshaped, y_train, epochs=100, learning_rate=0.01, Print=10)

# Evaluate the custom neural network model on the test set
custom_predictions = custom_model.predict(X_test_reshaped)
#custom_accuracy = np.mean((np.array(custom_predictions) > 0.5) == y_test)
custom_accuracy = np.mean((np.array(custom_predictions) > 0.5) == y_test.reshape(-1, 1, 1))
print("Custom Neural Network - Test Accuracy:", custom_accuracy)

# Convert Keras predictions to a DataFrame
keras_df_predictions = pd.DataFrame({'Actual': y_test.ravel(), 'Prediction': keras_model.predict(X_test_scaled).ravel()})
print("Keras Model Predictions:")
print(keras_df_predictions)

# Convert custom neural network predictions to a DataFrame
custom_df_predictions = pd.DataFrame({'Actual': y_test.ravel(), 'Prediction': np.array(custom_predictions).ravel()})
print("Custom Neural Network Predictions:")
print(custom_df_predictions)

# Create a bar chart comparing the accuracy of the two models
models = ['Keras', 'Custom Neural Network']
accuracies = [keras_accuracy, custom_accuracy]

plt.bar(models, accuracies)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies: Keras vs. Custom Neural Network')
plt.savefig('./results/accuracy_comparison_coupon.png')
plt.show()
