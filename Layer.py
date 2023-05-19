import numpy as np

class Layer:
    # Every layer in a neural network is treated as its own object with methods such as forward_propagation to compute the output of the layer
    # and backward propagation, to adjust the weights
    # x_size = number of input neurons
    # y_size = number of output neurons
    # activation = choose activation function for this layer, see available options below
    # Initializes one layer with 
    # Note that activation has to be chosen to be 'None' (no activation function), 'sigmoid' or 'tanh'

    #I start by defining the activation functions and their derivatives for backward propagation
    #they are not tight to any member of the class, but should always be available when the class is loaded, 
    #therefore I define them as static methods inside the Layer class

    @staticmethod
    def sigmoid(z):
        g = 1 / (1 + np.exp(-z))
        return (g)

    @staticmethod
    def Dsigmoid(z):
        g = np.exp(-z) / (1 + np.exp(-z))**2
        return (g)

    @staticmethod
    def Tanh(z):
        g = np.tanh(z)
        return g

    @staticmethod
    def DTanh(z):
        g = 1/np.cosh(z)**2
        return g
    
    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def Drelu(z):
        return np.where(z > 0, 1, 0)
    


    def __init__(self, x_size, y_size, activation):  # inialize layer with random weights
        self.weights = (np.random.rand(x_size, y_size) - 0.5)   # random weights between -0.5 and 0.5
        self.bias = (np.random.rand(1, y_size) - 0.5) # random biases between -0.5 and 1
        self.input = None    # a layer has information about the values that came from the previous layer
        self.output = None   # as well as information about its own values
        self.activation = activation
        self.shape = [x_size, y_size] # allows you to ask for the shape of the layer and its preceeding layer

    def reset(self):
        self.weights = (np.random.rand(self.shape[0], self.shape[1]) - 0.5)  # random weights between -0.5 and 0.5
        self.bias = (np.random.rand(1, self.shape[1]) - 0.5)  # random biases between -0.5 and 1

    #forward_propagation takes the input of the previous layer and computes the output of this layer
    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias # np.dot computes the matrix - vector product between the Transpose of the weight matrix and the input vector, this is in line with the 
        # notation in the lecture where weight wij refers to the i-ths node in the inputs layer and the j-th node in the output layer
        if self.activation == 'None':
            self.output = self.output
        elif self.activation == 'sigmoid':
            self.output = Layer.sigmoid(self.output)
        elif self.activation == 'tanh':
            self.output = Layer.Tanh(self.output)
        elif self.activation == 'relu':
            self.output = Layer.relu(self.output)
        else:
            print('Invalid activation function!')
        #self.output = self.output[0] # without this line I get shape incompatibilites, because the output is put into another unnecessary bracket and does not have the same shape as the input
        return self.output
    
    # Next: For backwards propagation we assume we know dE / dY - the matrix containing the derivatives of the error w.r.t. the layers output,
    #from this input we want to compute: the derivatives dE / dW and dE/dB of this layers weights and biases, this is used for updating the weights and biases in this layer
    # as well as dE / dX, the matrix containing the derivatives of the error w.r.t. the layers input. This can be used for the preceeding layer in backwards propagation.

    def backward_propagation(self, dEdY, learning_rate):
        output = np.dot(self.input, self.weights) + self.bias # this would be the output of this layer WITHOUT activation function
        # First I only back propagate the derivative w.r.t. activation function, not that the argument of the derivative is output
        if self.activation == 'None':
            dEdY = dEdY
        elif self.activation == 'sigmoid':
            dEdY = Layer.Dsigmoid(output)*dEdY # this is an element wise multiplication
        elif self.activation == 'tanh':
            dEdY = Layer.DTanh(output)*dEdY # this is an element wise multiplication
        elif self.activation == 'relu':
            dEdY = Layer.Drelu(output)*dEdY # this is an element wise multiplication
        else:
            print('Invalid activation function!')

        # Next I propagate the derivatives w.r.t. to weights matrix etc.
        # dEdW is the derivative w.r.t. to the weights / biases of that layer, which is needed for backpropagation
        
        dEdW = np.dot(np.transpose(self.input), dEdY)                 #np.dot(np.transpose(self.input), dEdY)
        dEdB = dEdY
        dEdX = np.dot(dEdY, np.transpose(self.weights))

        # update parameters
        self.weights -= learning_rate * dEdW
        self.bias -= learning_rate * dEdB

        return dEdX


