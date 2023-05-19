import numpy as np
from Layer import Layer
from NN import NN
import NNFactory
from sklearn.model_selection import KFold


distr1 = np.random.multivariate_normal(mean=[3,3,4], cov=[[2,2,2],[2,2,2],[2,2,2]], size=100, check_valid='raise')
# print(distr1)
distr2 = np.random.multivariate_normal(mean=[0,0,0], cov=[[1,1,1],[1,1,1],[1,1,1]], size=100, check_valid='raise')
# print(distr2)

labels = [[0] for i in range(100)]
distr1 = np.append(distr1, labels, axis=1)
labels = [[1] for i in range(100)]
distr2 = np.append(distr2, labels, axis=1)

data = np.append(distr1,distr2,axis=0)

indices = np.random.permutation(len(data))
data_shuffle = data[indices]

train_len = int(len(data_shuffle) * 0.8)
data_train = data_shuffle[0:train_len]
data_test = data_shuffle[train_len:len(data_shuffle)]

X_train = np.append(np.array(data_train[:,[[0]]]), np.array(data_train[:,[[1]]]), axis=2)
X_train = np.append(X_train, np.array(data_train[:,[[2]]]), axis=2)
y_train = np.array(data_train[:,[[3]]])

X_test = np.append(np.array(data_test[:,[[0]]]), np.array(data_test[:,[[1]]]), axis=2)
X_test = np.append(X_test, np.array(data_test[:,[[2]]]), axis=2)
y_test = np.array(data_test[:,[[3]]])

net = NNFactory.createNNByRandomizedSearch(X=X_train, y=y_train,
                                     range_learning_rate=[0.05, 0.15],
                                     range_layers=[2,2],
                                     range_epochs=[50,150],
                                     layer_size_ranges=[[2,2],[1,1]],
                                     act_fn_deep=['relu'],
                                     act_fn_output=['sigmoid'],
                                     cv=KFold(n_splits=4),
                                     n_iter=5,
                                     print_interval=10,
                                     verbose=2)

# len(net.layers) ... number of layers
# net.layers[0].shape ... shape of layer 1

print(net.predict(X_test))
print(y_test)




