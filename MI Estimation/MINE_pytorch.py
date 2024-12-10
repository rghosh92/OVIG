# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 16:41:58 2024

@author: User
"""
# import os
# path_orig = os.getcwd()
# os.chdir('C:\\Users\\User\\.spyder-py3\\VIG\\mine-pytorch')

import numpy as np
from mine import MINE
from sklearn.feature_selection import mutual_info_regression

def patch_asscalar(a):
    return a.item()

setattr(np, "asscalar", patch_asscalar)


def MINE_estimator(x_sample,y_sample,iters=2000,batch_size=400,hidden=32):
    
    
    # miEstimator = MINE(y_sample.shape[1],x_sample.shape[1], archSpecs={
    #     'layerSizes': [hidden] * 2,
    #     'activationFunctions': ['relu'] * 2
    # }, divergenceMeasure='KL', learningRate=1e-3)
    
    miEstimator = MINE(x_sample.shape[1],y_sample.shape[1], archSpecs={
        'layerSizes': [hidden] * 2,
        'activationFunctions': ['relu'] * 2
    }, divergenceMeasure='KL', learningRate=1e-3)

    ySamplesMarginal = np.random.permutation(y_sample)
    
    
    estimatedMI, estimationHistory = miEstimator.calcMI(x_sample, y_sample, x_sample, ySamplesMarginal,
                                                    batchSize=batch_size, numEpochs=iters)
    

    # statistics_network = nn.Sequential(
    #     nn.Linear(x_sample.shape[1]+y_sample.shape[1], 100),
    #     nn.ReLU(),
    #     nn.Linear(100, 100),
    #     nn.ReLU(),
    #     nn.Linear(100, 1)
    # )
    # mine = Mine(
    #     T = statistics_network,
    #     loss = 'mine', #mine_biased, fdiv
    #     method = 'concat',
    # )

    # mi = mine.optimize(x_sample, y_sample, iters = iters,batch_size=batch_size)
    return estimatedMI     


# os.chdir(path_orig)


# joint_samples = np.random.multivariate_normal(np.array([0,0]), np.array([[1, 1], [1, 1]]),size=800)

# X, Y = joint_samples[:, 0], joint_samples[:, 1]
# X = np.expand_dims(X,1)
# Y = np.expand_dims(Y,1)

