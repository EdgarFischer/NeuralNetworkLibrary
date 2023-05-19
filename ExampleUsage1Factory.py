import numpy as np
from Layer import Layer
from NN import NN
import NNFactory
from sklearn.model_selection import KFold

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = NNFactory.createNNByRandomizedSearch(X=x_train, y=y_train,
                                     range_learning_rate=[0.05, 0.15],
                                     range_layers=[2,2],
                                     range_epochs=[500,1500],
                                     layer_size_ranges=[[3,3],[1,1]],
                                     act_fn_deep=['sigmoid'],
                                     act_fn_output=['relu'],
                                     cv=KFold(n_splits=2),
                                     n_iter=5,
                                     print_interval=1000000,
                                     verbose=2)

# len(net.layers) ... number of layers
# net.layers[0].shape ... shape of layer 1

print(net.predict(x_train))




