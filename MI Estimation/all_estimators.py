# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 15:49:24 2024

@author: User
"""
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

import problexity as px
from MINE_pytorch import * 
from knnie import * 
from simplebinmi import * 
from normalize_functions import * 
import numpy as np 
from vig_lib import *
from MI_hybrid_generators import * 
# import mixed 

# from sklearn.feature_selection import mutual_info_classif

from npeet import entropy_estimators as ee

from variational_mi_lib import * 

class MI_Estimator():
    def __init__(self, params,mode='discrete'):
        self.params = params
        self.mode = mode
        
    def make_presentable(self,x_sample):
        x_sample = np.reshape(x_sample, (x_sample.shape[0], int(x_sample.size / x_sample.shape[0])))
        return x_sample
    
    
    def make_presentable_hot(self, y_sample):
        if self.mode == 'discrete':
            y_hot = np.zeros((y_sample.size, y_sample.max() + 1))
            y_hot[np.arange(y_sample.size), y_sample] = 1
            return y_hot
        else:
            y_sample = np.reshape(y_sample, (y_sample.shape[0], int(y_sample.size / y_sample.shape[0])))
            return y_sample 
    

    def MINE_MI(self,x_sample,y_sample):
        x_sample = self.make_presentable(x_sample)
        y_sample = self.make_presentable_hot(y_sample)
        
        MI =  MINE_estimator(x_sample,y_sample,iters = self.params[0],batch_size = self.params[1],hidden=self.params[2])
        return MI 
    
    def MINE_Global_MI(self,x_sample,y_sample):
        x_sample = self.make_presentable(x_sample)
        y_sample = self.make_presentable_hot(y_sample)
        
        normx = global_normalize(x_sample,C=1)
        normy = global_normalize(y_sample,C=1)
        MI =  MINE_estimator(normx,normy,iters = self.params[0],batch_size = self.params[1],hidden=self.params[2])
        return MI
        
    def MINE_Local_MI(self,x_sample,y_sample):
        x_sample = self.make_presentable(x_sample)
        y_sample = self.make_presentable_hot(y_sample)
        
        normx = local_normalize(x_sample,C=1)
        normy = local_normalize(y_sample,C=1)
        MI =  MINE_estimator(normx,normy,iters = self.params[0],batch_size = self.params[1],hidden=self.params[2])
        return MI
    
    def KSG(self,x_sample,y_sample):
        x_sample = self.make_presentable(x_sample)
        y_sample = self.make_presentable_hot(y_sample)
        
        MI = kraskov_mi(x_sample,y_sample,k=self.params[0])
        return MI 
    
    def KSG_local(self,x_sample,y_sample):
        x_sample = self.make_presentable(x_sample)
        y_sample = self.make_presentable_hot(y_sample)      
        
        C_z=self.params[1]
        MI_list = [] 
        
        for C_zi in C_z:
            normx = local_normalize(x_sample,C=C_zi)
            normy = local_normalize(y_sample,C=1)
            MI = kraskov_mi(normx,normy,k=self.params[0])
            MI_list.append(MI)
            
        Final_MI=max(MI_list)
        
        return MI 
        
    def KSG_global(self,x_sample,y_sample):
        x_sample = self.make_presentable(x_sample)
        y_sample = self.make_presentable_hot(y_sample)
        
        
        C_z=self.params[1]
        MI_list = [] 
        
        for C_zi in C_z:
            normx = global_normalize(x_sample,C=C_zi)
            normy = global_normalize(y_sample,C=1)
            MI = kraskov_mi(normx,normy,k=self.params[0])
            MI_list.append(MI)
            
        Final_MI=max(MI_list)
        return Final_MI 

        
    def KSG_revised(self,x_sample,y_sample):
        x_sample = self.make_presentable(x_sample)
        y_sample = self.make_presentable_hot(y_sample)

        
        MI = revised_mi(x_sample,y_sample,k=self.params[0],q=self.params[1])
        return MI 
    
    
    def M_VIG(self, x_sample, y_sample): 
        x_sample = self.make_presentable(x_sample)
        
        mvig = MVIG(x_sample, y_sample, hidden_ratio=self.params[0], batch_size=self.params[1], verbose=0, dataset_name='temp')
        return mvig
    
    def VInfo(self, x_sample, y_sample): 
        x_sample = self.make_presentable(x_sample)
        
        vi = VI(x_sample, y_sample, hidden_ratio=self.params[0], batch_size=self.params[1], verbose=0, dataset_name='temp')
        return vi
    
    def Mixed_KSG(self, x_sample,y_sample): 
        x_sample = self.make_presentable(x_sample)
        
        normx = local_normalize(x_sample,C=1)
        
        mixed_mi = ee.micd(normx, np.expand_dims(y_sample,axis=1), k=self.params[0],base=np.exp(1))
        return mixed_mi 
    
    # def Mixed_Mixed_KSG(self, x_sample,y_sample):
    #     x_sample = self.make_presentable(x_sample)
    #     y_sample = self.make_presentable_hot(y_sample)
        
    #     # normx = local_normalize(x_sample,C=1)
        
    #     mmi = mixed.Mixed_KSG(x_sample, y_sample,k=self.params[0])
        
    #     return mmi
    
    def infonce_(self, x_sample,y_sample):
        x_sample = self.make_presentable(x_sample)
        y_sample = self.make_presentable_hot(y_sample)
        # ['infonce', 'nwj', 'js', 'smile']
        batch_size = self.params[0]
        hidden_dim = self.params[1]
        layers = self.params[2]
        epochs = self.params[3]
        lr = self.params[4]

        mi = estimate_mi(x_sample.astype('float32'),y_sample.astype('float32'),'infonce',batch_size,hidden_dim,layers,epochs,lr)
        return mi
        
    def nwj_(self,x_sample,y_sample):
        x_sample = self.make_presentable(x_sample)
        y_sample = self.make_presentable_hot(y_sample)
        # ['infonce', 'nwj', 'js', 'smile']
        batch_size = self.params[0]
        hidden_dim = self.params[1]
        layers = self.params[2]
        epochs = self.params[3]
        lr = self.params[4]

        mi = estimate_mi(x_sample.astype('float32'),y_sample.astype('float32'),'nwj',batch_size,hidden_dim,layers,epochs,lr)
        return mi
    
    
    def js_(self,x_sample,y_sample):
        x_sample = self.make_presentable(x_sample)
        y_sample = self.make_presentable_hot(y_sample)
        # ['infonce', 'nwj', 'js', 'smile']
        batch_size = self.params[0]
        hidden_dim = self.params[1]
        layers = self.params[2]
        epochs = self.params[3]
        lr = self.params[4]

        mi = estimate_mi(x_sample.astype('float32'),y_sample.astype('float32'),'js',batch_size,hidden_dim,layers,epochs,lr)
        return mi
    
    
    def smile_(self,x_sample,y_sample):
        x_sample = self.make_presentable(x_sample)
        y_sample = self.make_presentable_hot(y_sample)
        # ['infonce', 'nwj', 'js', 'smile']
        batch_size = self.params[0]
        hidden_dim = self.params[1]
        layers = self.params[2]
        epochs = self.params[3]
        lr = self.params[4]
        mi = estimate_mi(x_sample.astype('float32'),y_sample.astype('float32'),'js',batch_size,hidden_dim,layers,epochs,lr,clip=self.params[5])
        return mi
    
    
    def get_norm(self,x_sample):
        return get_data_norm(x_sample)
    


if __name__ == "__main__":
    
    
    #  All Estimators are in nats
    
    mine_est = MI_Estimator([100,500,50],mode='continuous')
    KSG_est = MI_Estimator([3])
    
    hidden_ratio = np.linspace(0.1,1.0,num=10)
    batch_size = 200
    VI_est = MI_Estimator([hidden_ratio[-1],batch_size])
    
    
    joint_samples = np.random.multivariate_normal(np.array([0,0]), np.array([[1, 1], [1, 1]]),size=800)
    
    X, Y = joint_samples[:, 0], joint_samples[:, 1]
    X = np.expand_dims(X,1)
    Y = (Y>0.5).astype(int)
    print(Y)
    # print(VI_est.VInfo(X, Y))
    # print(KSG_est.KSG(X,Y))
        
    here = 1