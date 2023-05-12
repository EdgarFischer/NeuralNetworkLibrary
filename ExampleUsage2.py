import numpy as np
from Layer import Layer
from NN import NN

# training data
x_train = np.array([[[0,0,3]], [[0,1,2]], [[1,0,6]], [[1,1,2]]])
y_train = np.array([[[0,2]], [[1,4]], [[1,5]], [[0,8]]])

# network
net = NN()
net.addLayer(Layer(3, 5, 'tanh'))
net.addLayer(Layer(5, 2, 'None')) # since the output layer has values larger than 1, don't use an activation function here!

net.Train(x_train, y_train, epochs=1000, learning_rate=0.1, Print=500)

print(net.predict(x_train))





