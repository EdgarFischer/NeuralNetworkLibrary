import numpy as np
from Layer import Layer
from NN import NN
import NNFactory
from sklearn.model_selection import KFold

# training data
x_train = np.array([[['hello',13,45]],
                    [['there',14,'0x123']],
                    [['Maciej',13,45]],
                    [['there',12,0]]])

x_test = np.array([[['hello',13,45]],
                    [['Maciej',13,'0x123']],
                    [['unknown value',15,54]],
                    [['hello',15,0]]])

X_train_encoded = NNFactory.convertNominalFeatures(x_train, x_train)
print(X_train_encoded)

X_test_encoded = NNFactory.convertNominalFeatures(x_train, x_test)
print(X_test_encoded)



