import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import SGD
from NNFactory import NNFactory
from Layer import Layer
from NN import NN
from sklearn.neighbors import KNeighborsClassifier

# Import the custom neural network implementation
from NN import NN
from Layer import Layer

# Load the In-vehicle Coupon Recommendation dataset
coupon_data = pd.read_csv('./data/in-vehicle-coupon-recommendation.csv')

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

X_train_scaled = np.array(X_train_scaled).astype(np.float32)
y_train = np.array(y_train).astype(np.int32)
X_test_scaled = np.array(X_test_scaled).astype(np.float32)
y_test = np.array(y_test).astype(np.int32)

# Reshape the input data
X_train_reshaped = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
X_test_reshaped = X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1])

# Print the number of features and instances
num_features = X_train_reshaped.shape[2]
num_instances = X_train_reshaped.shape[0]
print("Number of features:", num_features)
print("Number of instances:", num_instances)

# Set custom LR for SGD optimizer to match the one from Custom Implementation
sgd_optimizer = SGD(learning_rate=0.01)

# Create the Keras model
def create_keras_model():
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the KerasClassifier
keras_model = KerasClassifier(build_fn=create_keras_model, epochs=50, batch_size=1, verbose=1)

# Perform cross-validation and calculate accuracy and recall
scoring = {'accuracy': make_scorer(accuracy_score), 'recall': make_scorer(recall_score)}
results = cross_validate(keras_model, X_train_scaled, y_train, cv=KFold(n_splits=5), scoring=scoring)

keras_cv_accuracy = np.mean(results['test_accuracy'])
keras_cv_recall = np.mean(results['test_recall'])

# Evaluate the Keras model on the test set
start_time = time.time()
keras_model.fit(X_train_scaled, y_train)
runtime_keras = time.time() - start_time

# Create the custom neural network model
model_custom = NNFactory.createNNByRandomizedSearch(
    X=X_train_reshaped,
    y=y_train,
    range_learning_rate=[0.001, 0.01],
    range_layers=[3, 3],
    range_epochs=[50, 50],
    layer_size_ranges=[[50, 100], [10, 50], [1, 1]],
    act_fn_deep=['sigmoid', 'tanh', 'relu'],
    act_fn_output=['sigmoid'],
    cv=KFold(n_splits=5),
    n_iter=10,
    print_interval=10,
    verbose=1
)

runtime_custom = model_custom.training_runtime
print(model_custom.training_runtime)

# Evaluate the custom neural network model on the test set
custom_predictions = model_custom.predict(X_test_reshaped)
custom_cv_accuracy = np.mean((np.array(custom_predictions) > 0.5) == y_test.reshape(-1, 1, 1))
# Calculate recall manually because why not
custom_true_positives = np.sum(np.logical_and((np.array(custom_predictions) > 0.5), y_test.reshape(-1, 1, 1)))
custom_false_negatives = np.sum(np.logical_and((np.array(custom_predictions) <= 0.5), y_test.reshape(-1, 1, 1)))
custom_cv_recall = custom_true_positives / (custom_true_positives + custom_false_negatives)

# Create the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)

# Fit the KNN model
start_time = time.time()
knn_model.fit(X_train_scaled, y_train)
runtime_knn = time.time() - start_time

# Perform cross-validation for KNN model
scoring = {'accuracy': make_scorer(accuracy_score), 'recall': make_scorer(recall_score)}
results = cross_validate(knn_model, X_train_scaled, y_train, cv=KFold(n_splits=5), scoring=scoring)

knn_cv_accuracy = np.mean(results['test_accuracy'])
knn_cv_recall = np.mean(results['test_recall'])

# Custom, Keras, and KNN accuracies
accuracies = [custom_cv_accuracy, keras_cv_accuracy, knn_cv_accuracy]
recalls = [custom_cv_recall, keras_cv_recall, knn_cv_recall]
runtimes = [runtime_custom, runtime_keras, runtime_knn]

# Plot the accuracy and recall in one subplot and runtimes in a separate subplot
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

accuracies = [float(acc) for acc in accuracies]
recalls = [float(rec) for rec in recalls]

# Accuracy and recall subplot
models = ['Custom', 'Keras', 'KNN']
x = np.arange(len(models))
width = 0.35
rects1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy')
rects2 = ax1.bar(x + width/2, recalls, width, label='Recall')
ax1.set_xlabel('Model')
ax1.set_ylabel('Score')
ax1.set_title('CV Accuracy and Recall Comparison: Custom vs Keras vs KNN')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylim(0, 1)
ax1.legend()

# Add exact values on top of each bar in the accuracy and recall subplot
for i, (v1, v2) in enumerate(zip(accuracies, recalls)):
    ax1.text(i - width/2, v1, f"Acc: {round(v1, 4)}", ha='center', va='bottom')
    ax1.text(i + width/2, v2, f"Rec: {round(v2, 4)}", ha='center', va='bottom')

plt.savefig('./results/crossval_acc_recall_runtime_comparison_coupon.png')
plt.show()
