import numpy as np
from Layer import Layer
from NN import NN

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = NN()
net.addLayer(Layer(2, 3, 'sigmoid'))
net.addLayer(Layer(3, 1, 'None'))  # since the output layer has values between -1 and 1 one can use the tanh activation function optionally

net.Train(x_train, y_train, epochs=1000, learning_rate=0.1, Print=10)

print(net.predict(x_train))