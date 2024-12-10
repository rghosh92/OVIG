# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 20:14:47 2024

@author: User
"""

""

import torch
from torch.utils import data
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from copy import deepcopy

from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.functional as F
import numpy as np
import sys, os
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import pickle
import random


class Dataset(data.Dataset):
    # Characterizes a dataset for PyTorch'
    def __init__(self, dataset_name, inputs, labels, transform=None, distractor=False, smoothing=False):
        # 'Initialization'
        self.labels = labels
        # self.list_IDs = list_IDs
        self.inputs = inputs
        self.smoothing = smoothing

        self.transform = transform
        self.distractor = distractor
        self.dataset_name = dataset_name
        # self.color_names = ['red','blue','green','yellow','violet','indigo','orange','purple','cyan','black']
        # self.color_class = []

        # for i in range(10):
        #     self.color_class.append(colors.to_rgb(self.color_names[i]))

    def __len__(self):
        # 'Denotes the total number of samples'
        return self.inputs.shape[0]



    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # ID = self.list_IDs[index]
        # Load data and get label
        # X = torch.load('data/' + ID + '.pt')
        img = self.inputs[index]

        # if self.dataset_name == 'STL10' or self.dataset_name == 'TINY_IMAGENET':
        #     img = np.transpose(img, [1, 2, 0])

        # Cutout module begins
        # xcm = int(np.random.rand()*95)
        # ycm = int(np.random.rand()*95)
        # img = self.cutout(img,xcm,ycm,24)
        #Cutout module ends

        # print(np.max(img),np.min(img))

        # img = Image.fromarray(np.uint8(img*255))

        # img = np.float32(scipy.misc.imresize(img, 2.0))
        # Optional:
        # img = img / np.max(img)

        # if self.distractor is True and self.labels[index] < 3:
        #     img = self.add_class_distractor(img,1,self.color_class[int(self.labels[index])])

        # if self.smoothing:
        #     img = gaussian_filter(img,sigma=(global_settings.global_SIGMA,global_settings.global_SIGMA,0))

        if self.transform is not None:
            img = self.transform(img)

        y = int(self.labels[index])

        return img, y

class Net_vanilla_NN_small(nn.Module):
    def __init__(self,input_channels,hidden_ratio=1.0,classes = 10):
        super(Net_vanilla_NN_small, self).__init__()


        layers = [int(50*hidden_ratio), int(50*hidden_ratio)]
        self.post_filter = False
        self.Temperature = 1
        # network layers
        self.conv1 = nn.Conv2d(input_channels, layers[0], 1)
        self.conv2 = nn.Conv2d(layers[0], layers[1], 1)


        self.bn1 = nn.BatchNorm2d(layers[0])
        self.bn2 = nn.BatchNorm2d(layers[1])
        
        self.relu = nn.ReLU()
        self.fc_direct = nn.Conv2d(layers[1], classes, 1)
        
    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.relu(x)
        x = self.bn1(x)
        # # print(x.shape)
        x = self.conv2(x)
        x = self.relu(x)
        # print(x.shape)
        x_checkpoint = self.bn2(x)
        # print(x.shape)
        # print(x.shape)
        # x_checkpoint = self.bn3(x)
        # # print(x.shape)
        # x = self.conv4(x)
        # # print(x.shape)
        # x = self.pool4(x)
        # # print(x.shape)
        # xm = self.bn4(x)

        # x = self.conv5(x)
        # x = self.pool5(x)
        # xm = self.bn5(x)
        # xm = self.bn3_mag(xm)
        # print(xm.shape)

        # xm = x_checkpoint.view(
            # [x_checkpoint.shape[0], x_checkpoint.shape[1] * x_checkpoint.shape[2] * x_checkpoint.shape[3], 1, 1])
        xm = self.fc_direct(x_checkpoint)/self.Temperature
        # xm = self.fc1(xm)
        # xm = self.relu(self.fc1bn(xm))
        # # xm = self.dropout(xm)
        # xm = self.fc2(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm
    
    

def generate_networks(input_channels, hidden_ratio, num_labels):
    Networks_to_train = [] 
    Networks_to_train_noise = [] 
    
    for i in range(len(hidden_ratio)):
        Networks_to_train.append(Net_vanilla_NN_small(input_channels,hidden_ratio[i],num_labels))
        Networks_to_train_noise.append(Net_vanilla_NN_small(input_channels,hidden_ratio[i],num_labels))
        
    return Networks_to_train, Networks_to_train_noise


def normalize_data(x_sample):
    x_sample_norm = (x_sample - x_sample.mean(axis=0)) / x_sample.std(axis=0)
    return x_sample
    # return x_sample_norm 

def generate_loaders(x_sample, y_sample,batch_size,dataset_name='temp'):
    
    x_sample = normalize_data(x_sample)
    x_sample = torch.from_numpy(x_sample).float().unsqueeze(2).unsqueeze(3)
    y_sample = torch.from_numpy(y_sample)
    
    my_dataset = Dataset(dataset_name, x_sample, y_sample)
    
    trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
    
    x_sample_copy = deepcopy(x_sample)
    y_sample_permuted = deepcopy(y_sample[torch.randperm(torch.numel(y_sample))])
    
    my_dataset_noise = Dataset(dataset_name, x_sample_copy, y_sample_permuted)
    
    trainloader_noise = torch.utils.data.DataLoader(my_dataset_noise, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
    
    return trainloader, trainloader_noise


def train_network_normal(net,trainloader, init_rate,total_epochs, weight_decay=0.0001):
    net = net
    net = net.cuda()
    net = net.train()
    optimizer = optim.SGD(net.parameters(), lr=init_rate, momentum=0.9, weight_decay=weight_decay)
    # optimizer = optim.Adam(net.parameters(), lr=0.0005, weight_decay=weight_decay)

#     scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma_learning)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], last_epoch=-1)

    criterion = nn.CrossEntropyLoss()
    
    init_epoch = 0
    all_train_losses = []
    for epoch in range(total_epochs):

        # print("Time for one epoch:",time.time()-s)
        # s = time.time()

        # scheduler.step()
        # print('epoch: ' + str(epoch))
        train_loss = []
        loss_weights = [] 

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            allouts = net(inputs)

            loss = criterion(allouts, labels.long())
            loss.backward()
            train_loss.append(loss.item())
            loss_weights.append(len(labels))
            
            
            optimizer.step()
            # print(0)
        
        all_train_losses.append(np.average(np.array(train_loss),weights=np.array(loss_weights)))
        
#         scheduler.step() 

        # print('break')
        
    # train_loss = train_loss/len(trainloader.sampler)
#     train_loss = np.mean(all_train_losses[-5:])
    train_loss = all_train_losses[-1]
    net = net.eval()

    return net,train_loss



def estimate_MVIG(x_sample, y_sample, Networks_to_train, Networks_to_train_noise, trainloader, trainloader_noise, training_params,verbose=1):
    
    
    diff_array = [] 
    normal_losses = []
    noise_losses = [] 
    
    for i in range(len(Networks_to_train)):
        # print(i)
        Networks_to_train[i],normal_loss = train_network_normal(Networks_to_train[i],trainloader, training_params.init_rate, training_params.total_epochs, training_params.decay_normal)
        Networks_to_train_noise[i],noise_loss = train_network_normal(Networks_to_train_noise[i],trainloader_noise, training_params.init_rate,training_params.total_epochs, training_params.decay_normal)
        
        diff_array.append(noise_loss-normal_loss)
        normal_losses.append(normal_loss)
        noise_losses.append(noise_loss)
    
    if verbose == 1: 
        
        print("noise:",noise_losses)
        print("normal:", normal_losses)
        print("diff:", diff_array)        
        print(np.max(diff_array))

    return np.max(diff_array)

def estimate_entropy(targets):
    num_labels = len(np.unique(targets))
    counts = np.zeros(num_labels)
    
    for i in range(len(targets)):
        counts[targets[i]] = counts[targets[i]] + 1
    ent = 0 
    counts = counts/sum(counts)
    for i in range(len(counts)):
        if counts[i] !=0:
            ent = ent + counts[i]*np.log(counts[i])
    ent = -ent
    return ent 



def estimate_VI(x_sample, y_sample, Network, trainloader, training_params,verbose=1):
    
    
    diff_array = [] 
    normal_losses = []
    noise_losses = [] 
    Network,normal_loss = train_network_normal(Network,trainloader, training_params.init_rate, training_params.total_epochs, training_params.decay_normal)
    ent = estimate_entropy(y_sample)
    
    
    if verbose == 1: 
        print('nothing here')

    return np.max(ent - normal_loss)


def MVIG(x_sample, y_sample, hidden_ratio, batch_size, verbose=1, dataset_name='temp'):

    input_channels = x_sample.shape[1]
    num_labels = len(np.unique(y_sample))
    trainloader, trainloader_noise = generate_loaders(x_sample, y_sample, batch_size)
    
    Networks_to_train,Networks_to_train_noise = generate_networks(input_channels, hidden_ratio, num_labels)
    
    training_params = type('', (), {})()
    training_params.init_rate = 0.03
    training_params.total_epochs = 200
    training_params.decay_normal = 0
    
    mvig = estimate_MVIG(x_sample, y_sample, Networks_to_train, Networks_to_train_noise, trainloader, trainloader_noise, training_params,verbose)

    return mvig 


def VI(x_sample,y_sample,hidden_ratio,batch_size,verbose=1,dataset_name='temp'):
    
    input_channels = x_sample.shape[1]
    num_labels = len(np.unique(y_sample))
    trainloader, trainloader_noise = generate_loaders(x_sample, y_sample, batch_size)
    
    Network = Net_vanilla_NN_small(input_channels,hidden_ratio,num_labels)
        
    training_params = type('', (), {})()
    training_params.init_rate = 0.03
    training_params.total_epochs = 200
    training_params.decay_normal = 0 
    
    estimated_vi = estimate_VI(x_sample, y_sample, Network, trainloader, training_params,verbose)
    
    return estimated_vi


