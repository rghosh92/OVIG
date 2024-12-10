# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 15:49:02 2024

@author: User
"""
import numpy as np 

def global_normalize(data, C):
    # Reshape data as before
    data = np.reshape(data, (data.shape[0], int(data.size / data.shape[0])))
    
    # Subtract the mean
    means = np.mean(data, axis=0)
    data = data - means
    
    # Compute the normalization factor
    norm = np.sqrt(np.mean(np.sum(data ** 2, axis=1)))
    
    # Normalize the data
    data *= C / norm
    
    return data

def local_normalize(data,C):
    data = np.reshape(data, (data.shape[0],int(data.size/data.shape[0])))
    
    means =  np.mean(data, axis=0) # find the mean for each dimension 
    data = data - means # data - means for each dimension
    
    norm = np.tile(np.sqrt(np.mean(data ** 2 ,axis=0)),(data.shape[0],1))
#     norm =  np.sqrt(np.mean(np.sum(sqz,axis=1)))
    normalized_data = C*data / (norm+(0.0000001))
    
    return normalized_data


def get_data_norm(data):
    
    data = np.reshape(data, (data.shape[0], int(data.size / data.shape[0])))
    
    # Subtract the mean
    means = np.mean(data, axis=0)
    data = data - means
    
    # Compute the normalization factor
    norm = np.sqrt(np.mean(np.sum(data ** 2, axis=1)))
    
    return norm 




