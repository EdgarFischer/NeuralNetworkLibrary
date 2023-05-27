import numpy as np

random_state = 42
np.random.seed(random_state)

import NNFactory
from Layer import Layer
from NN import NN
import time
import numpy

"""
This function creates a NN by random search. Either n_iter different NNs will be evaluated, or this whole thing
starts the last evaluation within ms_to_search milliseconds. Each parameter within the specified ranges will be chosen
with an equal probability (uniform distribution for learning rate, randint distribution otherwise)

:param X: Training data without labels
:param y: according labels to the training data
:param range_learning_rate: range of the learning rate for the search in the format [min, max].
:param range_epochs: range of number of epochs for the search in the format [min, max]. 
:param range_layers: range of number of layers for the search in the format [min, max]. 
:param layer_size_ranges: must be a list of ranges for all layers that might be possible in search. E.g. if you specify range_layers as [2,4], then this list must contain 4 ranges in the format [[min1,max1],[min2,max2],[min3,max3],[min4,max4]]
:param act_fn_deep: list of activation functions for all layers except the last one to search. Must be a subset of ['None', 'sigmoid', 'tanh', 'relu']
:param act_fn_output: list of output activation functions to search. Must be a subset of ['None', 'sigmoid', 'tanh', 'relu']
:param cv: must be an instance of a sklearn.model_selection Cross validator! E.g. use KFold, StratifiedKFold, etc etc.
:param n_iter: number of evaluations. When using this, don't specify ms_to_search 
:param s_to_search: time within the last evaluation must start. When using this, don't specify n_iter
:param print_interval: every print_interval training epoch will be printed in the console. Default 10.
:param verbose: 0: print nothing, 1: print infos before each search iteration. 2: print infos about each CV Fold. Default 1

:returns the NN with the best parameters, already trained with the entire training data (X)
"""
@staticmethod
def createNNByRandomizedSearch(X, y, range_learning_rate, range_epochs, range_layers, layer_size_ranges, act_fn_deep, act_fn_output, cv, n_iter=None, s_to_search=None, print_interval=10, verbose=1):
    if n_iter is None and s_to_search is None:
        raise Exception('at least n_iter or ms_to_search must be given!')
    if n_iter is not None and s_to_search is not None:
        raise Exception('n_iter and ms_to_search cannot be specified both')
    if range_layers[0]<1:
        raise Exception('The minimum of layers must be 1!')
    if range_layers[1]>1 and act_fn_deep is None:
        raise Exception("You must specify a deep activation function range if the layers can be more than 1!")
    index = 0
    starting_time = time.time()

    best_network = None
    best_mse = np.inf

    # Loop that does one randomized search in iteration
    while True:

        # Picking randomized parameters according to the specified ranges
        learning_rate = numpy.random.uniform(range_learning_rate[0], range_learning_rate[1])
        epochs = numpy.random.randint(range_epochs[0], range_epochs[1]+1)
        layers = numpy.random.randint(range_layers[0], range_layers[1]+1)
        fn_deep = act_fn_deep[numpy.random.randint(0, len(act_fn_deep))] if act_fn_deep is not None else None
        fn_output = act_fn_output[numpy.random.randint(0, len(act_fn_output))]

        # Code that constructs our Neural Network that is later used in the cross validation
        net = NN()
        previousSize = X.shape[2]
        layersStr = "{}in".format(previousSize)
        # Loop that adds the layers to the NN
        for l in range(layers):
            layersize = numpy.random.randint(layer_size_ranges[l][0], layer_size_ranges[l][1]+1)
            layerfn = fn_deep if l < (layers-1) else fn_output
            net.addLayer(Layer(previousSize, layersize, layerfn))
            layersStr = layersStr + " - {}{}".format(layersize, layerfn)
            previousSize = layersize

        average_mse = 0
        if verbose>=1:
            print("Search Iteration {}: LR={}, epochs={}, layers=({})".format(index+1, learning_rate, epochs, layersStr))

        # Loop that iterates for each split in the cross validation
        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            if verbose>=2:
                print("Checking fold {}/{}:".format(i+1, cv.get_n_splits()))
            X_fold = X[np.array(train_index)]
            y_fold = y[np.array(train_index)]
            net.Train(X_fold, y_fold, epochs=epochs, learning_rate=learning_rate, Print=print_interval)
            y_pred = net.predict(X[np.array(test_index)])
            y_true = y[np.array(test_index)]
            fold_mse = NNFactory.getMSEForPredictions(y_pred, y_true)
            if verbose>=2:
                print("Fold MSE result: {}".format(fold_mse))
            # calculate average MSE for the CV
            average_mse = fold_mse if i == 0 else ((i*average_mse) + fold_mse)/(i+1)

            # Reset the layers of the NN in order to re-train it for the next split
            net.resetLayers()

        if verbose>=1:
            print("Search iteration completed! Result MSE = {}\n".format(average_mse))

        # If we have found a new best network, then reset it, train it with the entire training data, and save it for later return
        if average_mse < best_mse:
            net.resetLayers()
            net.Train(X, y, epochs=epochs, learning_rate=learning_rate, Print=print_interval)
            best_network = net
            best_mse = average_mse

        # Checking search stop conditions
        index = index + 1
        if index == n_iter:
            break
        if s_to_search is not None and (time.time() - starting_time) >= s_to_search:
            break

    return best_network


"""
This function converts non-numerical features in the dataset each to a set of one-hot-encoded binary features and returns
the resulting dataset.

:param X_ref: the dataset to use as a reference. From this dataset we find out where the nominal attributes are and how many distinct values each attribute has
:param X_toconvert: the dataset to convert (encode). Must have the same shape as X_ref
:returns X_toconvert if X_toconvert or X_ref only contain numerical features, otherwise, if X has n columns, a dataset with n + sum(m[i]-1) columns, whereas
         for each non-numeric column (there are i of them), m[i] is the number if distinct non-numeric values in the ith non-numeric column. 
"""
@staticmethod
def convertNominalFeatures(X_ref, X_toconvert):
    bools = None
    try:
        np.char.isnumeric(X_toconvert)
        bools = np.char.isnumeric(X_ref)
    except TypeError:
        # No non-numerical features in there!
        return X_toconvert

    # Find the nominal attributes in the reference dataset and prepare everything so X_toconvert can be encoded
    # Find out at what column indices there are nominal attributes and sort it
    nom_indices = []
    for i in range(len(bools)):
        for j in range(len(bools[0][0])):
            if not bools[i][0][j]:
                nom_indices.append(j)
    nom_indices_unique = np.sort(np.unique(np.array(nom_indices)))
    # In the same order as the column indices, create an 2-dim array containing the distinct nominal values for each attribute
    unique_noms = []
    for unique_ind in nom_indices_unique:
        unique_noms.append(np.unique(X_ref[:, 0, unique_ind]).tolist())

    # Now encode the dataset to convert accordingly
    new_X = None
    for i in range(len(X_toconvert)):
        new_row = np.array([])
        running_index = 0
        for j in range(len(X_toconvert[0][0])):
            if j in nom_indices_unique:
                noms = unique_noms[running_index]
                running_index += 1
                new_row = np.append(new_row, [1 if (X_toconvert[i][0][j] == nom) else 0 for nom in noms])
            else:
                new_row = np.append(new_row, X_toconvert[i][0][j])
        if new_X is None:
            new_X = np.array([[new_row]])
        else:
            new_X = np.append(new_X, [[new_row]], axis=0)

    return new_X


# Helper that returns the MSE for two sets of labels. Is used to score the CV-folds in the randomized search
@staticmethod
def getMSEForPredictions(y_pred, y_true):
    MSE = 0  # Mean squared error
    for i in range(len(y_pred)):
        MSE += NN.mse(y_pred[i], y_true[i])
    MSE /= len(y_pred)
    return MSE