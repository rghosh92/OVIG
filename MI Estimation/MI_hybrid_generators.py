# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 20:52:13 2024

@author: User
"""
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import pickle 
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


class Hybrid_Generator():
    def __init__(self, means, covs,probs='uniform'):
        self.means = means
        self.covs = covs
        self.num_labels = len(self.means)
        self.probs = np.zeros(self.num_labels)
        if probs =='uniform':
            for i in range(self.num_labels):
                self.probs[i] = 1/float(self.num_labels)   
        else:
            self.probs = probs 
            
        self.mv_generators = []
        for i in range(self.num_labels):
            self.mv_generators.append(multivariate_normal(self.means[i], self.covs[i]))
    
    def generate_samples(self, number):
        x_samples = []
        y_samples = [] 
        for i in range(number):
            y_sample = np.random.randint(self.num_labels)
            x_sample = self.mv_generators[y_sample].rvs()
            x_samples.append(x_sample)
            y_samples.append(y_sample)
            
        return x_samples,y_samples
    

def equicovariant_Gaussian_multiclass_distribution(num_classes,dim,radial_var=1,cov_multiplier=1):
    
    means = np.random.multivariate_normal(np.zeros(dim), np.identity(dim)*radial_var,size=num_classes)
    # print(means)
    covs = []
    A = np.random.rand(dim, dim)
    cov_mat = np.dot(A,A.transpose())*cov_multiplier
    result = None
    while result is None:
        try:
            for i in range(num_classes):
                multivariate_normal(means[i], cov_mat)
            result = 0
        except:
            A = np.random.rand(dim, dim)
            cov_mat = np.dot(A,A.transpose())*cov_multiplier
            covs = []
            for i in range(num_classes):
                covs.append(cov_mat) 
            pass

    for i in range(num_classes):
        covs.append(cov_mat)    
    # print(covs)
    return means, covs


def random_Gaussian_multiclass_distribution(num_classes,dim,radial_var=1,cov_multiplier=1):
    
    means = np.random.multivariate_normal(np.zeros(dim), np.identity(dim)*radial_var,size=num_classes)
    # print(means)
    covs = []
    
    for i in range(num_classes):
        A = np.random.rand(dim, dim)
        cov_mat = np.dot(A,A.transpose())*cov_multiplier
        covs.append(cov_mat) 
        
    result = None
    while result is None:
        try:
            for i in range(num_classes):
                multivariate_normal(means[i], covs[i])
            result = 0
        except:
            for i in range(num_classes):
                A = np.random.rand(dim, dim)
                cov_mat = np.dot(A,A.transpose())*cov_multiplier
                covs.append(cov_mat)  
            pass

       
    # print(covs)
    return means, covs

    


def MCMC_MI(hybrid_gen,x_samples=None, y_samples=None, num_samples=1000):
    if x_samples == None:
        x_samples, y_samples = hybrid_gen.generate_samples(num_samples)
    else:
        num_samples = len(x_samples)
    print(np.mean(np.abs(x_samples)))
    # print(np.min(x_samples))
    pmi_sum = 0 
    for i in range(num_samples):
        P_X_given_Y = np.zeros(hybrid_gen.num_labels)
        for j in range(hybrid_gen.num_labels):
            P_X_given_Y[j] = hybrid_gen.mv_generators[j].pdf(x_samples[i])
        P_X = np.dot(P_X_given_Y,hybrid_gen.probs)
        pmi_sum += np.log(P_X_given_Y[y_samples[i]]/P_X)
    return pmi_sum/num_samples
        

if __name__ == "__main__":
    
    total_trials = 100
    MI_list = []
    means_list = []
    covs_list = []
    num_classes = 2 
    dim = 100 
    radial_var = 0.1
    
    for i in range(total_trials):
        # print(i)
        if np.mod(i,2)==1:
            means,covs = equicovariant_Gaussian_multiclass_distribution(num_classes=num_classes, dim = dim,radial_var=0)
        else:
            means,covs = equicovariant_Gaussian_multiclass_distribution(num_classes=num_classes, dim = dim,radial_var=radial_var)

        # means,covs = random_Gaussian_multiclass_distribution(num_classes=num_classes, dim = dim,radial_var=radial_var)
        
        prob_gen = Hybrid_Generator(means,covs)
        # x_samples, y_samples = prob_gen.generate_samples(1000)
        MI = MCMC_MI(prob_gen,num_samples=1000)
        print(MI)
        MI_list.append(MI)
        means_list.append(means)
        covs_list.append(covs)
    
    
    save_name = 'MI_Est_Experiments_num_classes_' + str(num_classes) +'_Dim_'+ str(dim)+ '_radial_var_'+str(radial_var)+'extreme_configs.pkl'
    file_dir = os.path.join(dir_path, save_name)         
    
    
    with open(file_dir, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([means_list,covs_list,MI_list], f)
    
    
    print(np.std(np.array(MI_list)))
    plt.hist(MI_list,bins=20)
    plt.show()

#  Settings: 
#  means,covs = equicovariant_Gaussian_multiclass_distribution(num_classes=2, dim = 100,radial_var=0.001,cov_multiplier=1)
#  means,covs = equicovariant_Gaussian_multiclass_distribution(num_classes=2, dim = 10,radial_var=0.005,cov_multiplier=1)
#  means,covs = equicovariant_Gaussian_multiclass_distribution(num_classes=2, dim = 3,radial_var=0.03,cov_multiplier=1)
#  means,covs = random_Gaussian_multiclass_distribution(num_classes=2, dim = 3,radial_var=0.002,cov_multiplier=1)


    
