import numpy as np
from Layer import Layer

class NN:

    # For backward propagation I need the mse and its derivative w.r.t. Y. These functions should always be available in this class
    # and are hence added as static methods
    @staticmethod
    def mse(y_pred, y_true):  # this method computes the mse
        g = np.mean(np.power(y_true-y_pred, 2))
        return g

    @staticmethod
    def Dmse(y_pred, y_true):  # this method computes the dEdY, E refers to the mse
        g = 2*(y_pred-y_true)/y_true.size
        return g

    # a neural network (NN) is simply a collection of its layers, it is always created with 0 layers, and layers can be added with the addLayer function
    def __init__(self):
        self.layers = []

    # this adds a layer to the NN
    def addLayer(self, layer):
        self.layers.append(layer)

    # Resets weights and biases of all layers to a random value
    def resetLayers(self):
        for layer in self.layers:
            layer.reset()

    # The next function predicts the outcome for a NN for a an arbitrary number of samples
    def predict(self, input):
        N = len(input)

        prediction = []
        # go through all samples
        for i in range(N):
            # forward propagation
            output = input[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            prediction.append(output)

        return prediction

    # Lastly, I want to use stochastic gradient decent to train the network
   
    def Train(self, x_train, y_train, epochs, learning_rate, Print): #the Print variable means only print information after every Print number of epochs
        # number of training instances = N
        N = len(x_train)

        # loop for epochs
        for i in range(epochs):
            # Shuffle the training data before every epoch, to add more randomness
            indices = np.random.permutation(N)
            x_train_shuffle = x_train[indices]
            y_train_shuffle = y_train[indices]

            MSE = 0 # Mean squared error

            for j in range(N):
                # forward propagation
                output = x_train_shuffle[j]
                for layer in self.layers: # note that this loop cannot be avoided with the predict function, because the layers need to save the new input / output values
                    output = layer.forward_propagation(output)

                # compute error for this sample
                MSE += NN.mse(output, y_train_shuffle[j]) 

                # backward propagation
                dEdY = NN.Dmse(output, y_train_shuffle[j])  # derivative of mse w.r.t. Y for the very last layer.
                for layer in reversed(self.layers):
                    dEdY = layer.backward_propagation(dEdY, learning_rate)

            # calculate average error on all samples
            MSE /= N
            if (i+1) % Print == 0:
                print('epoch: ' + str(i+1)+ ', MSE: '+str(MSE))  #print('epoch %d/%d   MSE=%f' % (i+1, epochs, MSE))




