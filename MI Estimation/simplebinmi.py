# Simplified MI computation code from https://github.com/ravidziv/IDNNs
import numpy as np

def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse

from scipy.stats import entropy

def bin_calc_information_new_mod(inputdata, layerdata, num_of_bins_input=5, num_of_bins_layer=5):
    input_min = np.min(inputdata)
    input_max = np.max(inputdata)
    bin_input = np.linspace(input_min, input_max, num_of_bins_input, dtype='float32')
    digitized_input = bin_input[np.digitize(np.squeeze(inputdata.reshape(1, -1)), bin_input)- 1].reshape(len(inputdata), -1)
    #     print(digitized_input)
    layer_min = np.min(layerdata)
    layer_max = np.max(layerdata)
    bin_layer = np.linspace(layer_min, layer_max, num_of_bins_layer, dtype='float32')
    digitized = bin_layer[np.digitize(np.squeeze(layerdata.reshape(1, -1)), bin_layer) - 1].reshape(len(layerdata), -1)
    #     print(digitized)
    digitized_concat = np.concatenate((digitized_input, digitized), axis=1)
    #     print(digitized_concat)
    value, counts_input = np.unique(digitized_input, return_counts=True, axis=1)
    value, counts_layer = np.unique(digitized, return_counts=True, axis=1)
    value, counts_concat = np.unique(digitized_concat, return_counts=True, axis=1)
    return entropy(counts_input, base=2) + entropy(counts_layer, base=2) - entropy(counts_concat, base=2)

