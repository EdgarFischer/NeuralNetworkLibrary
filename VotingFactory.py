import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers.legacy import SGD
from NNFactory import NNFactory
from Layer import Layer
from NN import NN
import matplotlib.pyplot as plt
import time

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
missing_values = ['unknown']
X_train = X_train.replace(missing_values, np.nan)
X_test = X_test.replace(missing_values, np.nan)

# Impute missing values with the most frequent value in each column
imputer = SimpleImputer(strategy='most_frequent')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Labeling target categorical data
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# One-Hot Encoding for y/n input
onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_encoded = onehot_encoder.fit_transform(X_train)
X_test_encoded = onehot_encoder.transform(X_test)

# Apply scaling to the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Convert to NumPy arrays
X_train_scaled = np.array(X_train_scaled)
y_train = np.array(y_train)
X_test_scaled = np.array(X_test_scaled)
y_test = np.array(y_test)

# Reshape X_train and X_test to have shape (num_samples, 1, num_features)
X_train_reshaped = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
X_test_reshaped = X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1])

# Set custom LR for SGD optimizer to match the one from Custom Implementation
sgd_optimizer = SGD(learning_rate=0.009053152647299674)

# Define the Keras model builder function
# Parameters are set to match the best params for custom implementation
def build_keras_model():
    model = Sequential()
    model.add(Dense(67, input_dim=X_train_scaled.shape[1], activation='tanh'))
    model.add(Dense(14, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

# Build the KerasClassifier
model_keras = KerasClassifier(build_fn=build_keras_model, epochs=30, batch_size=1)

# Evaluate the Keras model using cross-validation
scoring = {'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score)}
results = cross_validate(model_keras, X_train_scaled, y_train, cv=KFold(n_splits=5), scoring=scoring)

keras_cv_accuracy = np.mean(results['test_accuracy'])
keras_cv_precision = np.mean(results['test_precision'])

# Training on holdout for runtime measurements
start_time = time.time()
model_keras.fit(X_train_scaled, y_train)
runtime_keras = time.time() - start_time

# Build and train the neural network using custom NN class
model_custom = NNFactory.createNNByRandomizedSearch(
    X=X_train_reshaped,
    y=y_train,
    range_learning_rate=[0.001, 0.01],
    range_layers=[3, 3],
    range_epochs=[30, 30],
    layer_size_ranges=[[50, 100], [10, 50], [1, 1]],
    act_fn_deep=['sigmoid', 'tanh'],
    act_fn_output=['sigmoid'],
    cv=KFold(n_splits=5),
    n_iter=10,
    print_interval=10,
    verbose=1
)

runtime_custom = model_custom.training_runtime

# Evaluate the custom model
predictions_custom = model_custom.predict(X_test_reshaped)
rounded_predictions_custom = np.round(predictions_custom).astype(int)
y_test_reshaped = y_test.reshape(rounded_predictions_custom.shape)
custom_cv_accuracy = np.mean(rounded_predictions_custom == y_test_reshaped)
# Calculate the precision by comparing the positive predictions with the true positive labels
custom_cv_precision = np.mean(rounded_predictions_custom[rounded_predictions_custom == 1] == y_test_reshaped[rounded_predictions_custom == 1])



# Create the KNN model
model_knn = KNeighborsClassifier(n_neighbors=5)

start_time_knn = time.time()
# Train the KNN model
model_knn.fit(X_train_scaled, y_train)
end_time_knn = time.time()
runtime_knn = end_time_knn - start_time_knn

# Evaluate the KNN model using cross-validation
# Define the scoring metrics
scoring = {'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score)}

# Perform cross-validation and calculate accuracy and precision
# knn_cv_accuracy = np.mean(cross_val_score(model_knn, X_train_scaled, y_train, cv=5, scoring='accuracy'))
# knn_cv_precision = np.mean(cross_val_score(model_knn, X_train_scaled, y_train, cv=5, scoring='precision'))

scoring = {'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score)}
results = cross_validate(model_knn, X_train_scaled, y_train, cv=KFold(n_splits=5), scoring=scoring)

knn_cv_accuracy = np.mean(results['test_accuracy'])
knn_cv_precision = np.mean(results['test_precision'])

# Add accuracies and precisions to the lists
accuracies = [custom_cv_accuracy, keras_cv_accuracy, knn_cv_accuracy]
precisions = [custom_cv_precision, keras_cv_precision, knn_cv_precision]

# Add runtimes to the list
runtimes = [runtime_custom, runtime_keras, runtime_knn]

# Create the bar chart for accuracy and precision
models = ['Custom', 'Keras', 'KNN']
# Create the bar chart for accuracy and precision, along with runtime
fig, ax = plt.subplots(figsize=(10, 6))

# Plot accuracy and precision
bar_width = 0.35
opacity = 0.8
index = np.arange(len(models))

ax.bar(index, accuracies, bar_width, alpha=opacity, label='Accuracy')
ax.bar(index + bar_width, precisions, bar_width, alpha=opacity, label='Precision')
ax.set_xlabel('Model')
ax.set_ylabel('Scores')
ax.set_title('CV Accuracy and Precision Comparison: Custom vs Keras vs KNN')
ax.set_xticks(index + bar_width/2)
ax.set_xticklabels(models)
ax.set_ylim(0, 1)
ax.legend()

# Add exact values on top of each bar
for i, (v, p, r) in enumerate(zip(accuracies, precisions, runtimes)):
    ax.text(i, v, f"Acc: {round(v, 4)}\nTime: {round(r, 2)}s", ha='center', va='bottom')
    ax.text(i + bar_width, p, f"Prec: {round(p, 4)}\nTime: {round(r, 2)}s", ha='center', va='bottom')

plt.savefig('./results/crossval_acc_prec_runtime_comparison_voting.png')
plt.show()
