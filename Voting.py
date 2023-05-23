import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from Layer import Layer
from NN import NN

# Load the dataset
data = pd.read_csv('./data/CongressionalVotingID.shuf.lrn.csv')

# Preprocess the data
# Remove irrelevant columns
if 'id' in data.columns:
    data = data.drop('id', axis=1)

# Separate features and target variable
X = data.drop('class', axis=1)
y = data['class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling missing values
# Identify missing values (assuming they are represented as '?' in this dataset)
missing_values = ['?']
X_train = X_train.replace(missing_values, np.nan)
X_test = X_test.replace(missing_values, np.nan)

# Impute missing values with the most frequent value in each column
imputer = SimpleImputer(strategy='most_frequent')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Labeling categorical data
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# One-Hot Encoding for categorical features
onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_encoded = onehot_encoder.fit_transform(X_train)
X_test_encoded = onehot_encoder.transform(X_test)

# Apply scaling to the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Reshape X_train and X_test to have shape (num_samples, 1, num_features)
X_train_reshaped = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
X_test_reshaped = X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1])

# Build and train the neural network using Keras
model_keras = Sequential()
model_keras.add(Dense(100, input_dim=X_train_scaled.shape[1], activation='relu'))
model_keras.add(Dense(50, activation='relu'))
model_keras.add(Dense(1, activation='sigmoid'))

model_keras.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_keras.fit(X_train_scaled, y_train, epochs=100, batch_size=16)

# Evaluate the Keras model
loss_keras, accuracy_keras = model_keras.evaluate(X_test_scaled, y_test)
print("Accuracy (Keras):", accuracy_keras)

# Build and train the neural network using custom NN class
model_custom = NN()
model_custom.addLayer(Layer(X_train_scaled.shape[1], 100, activation='relu'))
model_custom.addLayer(Layer(100, 50, activation='relu'))
model_custom.addLayer(Layer(50, 1, activation='sigmoid'))

model_custom.Train(X_train_reshaped, y_train, epochs=100, learning_rate=0.001, Print=10)

# Evaluate the custom model
predictions_custom = model_custom.predict(X_test_reshaped)
rounded_predictions_custom = np.round(predictions_custom).astype(int)
accuracy_custom = np.mean(rounded_predictions_custom == y_test)
print("Accuracy (Custom):", accuracy_custom)

import matplotlib.pyplot as plt

# Create a bar chart comparing accuracies
models = ['Custom', 'Keras']
accuracies = [accuracy_custom, accuracy_keras]

plt.bar(models, accuracies)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: Custom vs Keras')
plt.ylim(0, 1)  # Set the y-axis limit to range from 0 to 1

# Add exact values on top of each bar
for i, v in enumerate(accuracies):
    plt.text(i, v, str(round(v, 4)), ha='center', va='bottom')

plt.savefig('./results/accuracy_comparison_voting.png')
plt.show()
