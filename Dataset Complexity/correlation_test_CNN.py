

"""
Created on Thu May 23 13:35:19 2024

@author: User
"""


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


class ConvolutionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super(ConvolutionLayer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (int(padding), int(padding))
        self.conv = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, stride=self.stride,
                              padding=self.padding)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x)

class Net_vanilla_cnn_small(nn.Module):
    def __init__(self,input_channels,hidden_ratio=1.0,classes = 10):
        super(Net_vanilla_cnn_small, self).__init__()

        kernel_sizes = [5, 3, 3]
        pads = (np.array(kernel_sizes) - 1) / 2
        pads = pads.astype(int)
        layers = [int(30*hidden_ratio), int(60*hidden_ratio),int(60*hidden_ratio)]
        self.post_filter = False
        # network layers
        self.conv1 = ConvolutionLayer(input_channels, layers[0], [kernel_sizes[0], kernel_sizes[0]], stride=1, padding=pads[0])
        self.conv2 = ConvolutionLayer(layers[0], layers[1], [kernel_sizes[1], kernel_sizes[1]], stride=1,
                                      padding=pads[1])
        # self.conv5 = ConvolutionLayer(layers[3], layers[4], [kern1el_sizes[2], kernel_sizes[2]], stride=1, padding=pads[2])

        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(layers[0])
        self.pool2 = nn.MaxPool2d(kernel_size=(8, 8))
        self.bn2 = nn.BatchNorm2d(layers[1])
        # self.pool5 = nn.MaxPool2d(2)
        # self.bn5 = nn.BatchNorm2d(layers[4])
        self.fc1 = nn.Conv2d(layers[1], layers[2], 1)
        self.fc1bn = nn.BatchNorm2d(layers[2])
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout2d(0.7)
        # self.fc2 = nn.Conv2d(layers[2], 10, 1)
        self.fc_direct = nn.Conv2d(layers[1], classes, 1)
        
    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool2(x)
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

        xm = x_checkpoint.view(
            [x_checkpoint.shape[0], x_checkpoint.shape[1] * x_checkpoint.shape[2] * x_checkpoint.shape[3], 1, 1])
        xm = self.fc_direct(xm)
        # xm = self.fc1(xm)
        # xm = self.relu(self.fc1bn(xm))
        # # xm = self.dropout(xm)
        # xm = self.fc2(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm

    


# +
def train_network_normal(net,trainloader, init_rate,total_epochs, step_size, gamma_learning, weight_decay=0.0001):
    net = net
    net = net.cuda()
    net = net.train()
#     optimizer = optim.SGD(net.parameters(), lr=init_rate, momentum=0.9, weight_decay=weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=weight_decay)

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
            # inputs = inputs.cuda()
            # labels = labels.cuda()

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
# -



# +
def train_network_noise(net,trainloader, init_rate,total_epochs,step_size,gamma_learning, weight_decay):
    net = net
    net = net.cuda()
    net = net.train()
#     optimizer = optim.SGD(net.parameters(), lr=init_rate, momentum=0.9, weight_decay=weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=weight_decay)

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
            # inputs = inputs.cuda()
            # labels = labels.cuda()

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
    train_loss = all_train_losses[-1]
    
    net = net.eval()

    return net,train_loss
# -
from copy import copy

def rand_another(label,label_max):
    array = torch.arange(label_max+1)
        
    array_removed = torch.cat([array[0:label], array[label+1:]]) 
    return array_removed[np.random.randint(0,len(array_removed))]


def estimate_entropy(targets,dataset_name):
    
    if dataset_name == 'MNIST':
        counts = np.zeros(10)
    elif dataset_name == 'CIFAR10':
        counts = np.zeros(10)
    elif dataset_name == 'CIFAR100':
        counts = np.zeros(100)
    
    for i in range(len(targets)):
        counts[targets[i]] = counts[targets[i]] + 1
    ent = 0 
    counts = counts/sum(counts)
    for i in range(len(counts)):
        if counts[i] !=0:
            ent = ent + counts[i]*np.log(counts[i])
    ent = -ent
    return ent 

# +
def estimate_yhat_given_y_entropy(dataset_name,noise_prob):
    
    if dataset_name == 'MNIST':
        num_classes = 10 
    elif dataset_name == 'CIFAR10':
        num_classes = 10 
    elif dataset_name == 'CIFAR100':
        num_classes = 100 
        
        
    counts = np.zeros(num_classes)
    counts[0] = 1-noise_prob
    for i in range(1,len(counts)):
        counts[i] = (noise_prob)/(num_classes - 1) 
    
    ent = 0 
    for i in range(len(counts)):
        if counts[i] !=0:
            ent = ent + counts[i]*np.log(counts[i])
    ent = -ent
    return ent

def test_network(net, testloader, test_labels):
    net = net.eval()
    criterion = nn.CrossEntropyLoss()
    correct = torch.tensor(0)
    dataiter = iter(testloader)
    test_losses = [] 
    loss_weights = [] 
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs, labels = data
            
            all_outs = net(inputs)
            predicted = torch.argmax(all_outs,1)
            loss = criterion(all_outs, labels.long())
            test_losses.append(loss.item())
            loss_weights.append(len(labels))
            correct = correct + torch.sum(predicted == labels)
            
        test_loss = np.average(np.array(test_losses),weights=np.array(loss_weights))
    accuracy = float(correct) / float(len(test_labels))
    print("acc:", accuracy)
    print("tloss:", test_loss)
    return accuracy, test_loss



# -

class mini_dataset:
    data = 0 
    targets = 0 


# +
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("num_train_labels")
parser.add_argument("total_trials")
parser.add_argument("variable_datasize")
args = parser.parse_args()



if __name__ == "__main__":
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # --------------- Hyperparams for MNIST------------------#
    dataset_name = args.dataset
    if args.dataset == 'MNIST':
        input_channels = 1
    else:
        input_channels = 3
        
    num_labels = 10 
    if args.dataset == 'CIFAR100':
        num_labels = 100
    num_train_labels = int(args.num_train_labels)
    total_trials = int(args.total_trials)
    label_list = np.arange(0,num_labels,1)
    variable_datasize = args.variable_datasize

    
    training_size_total = 50000
    
    if args.dataset == 'CIFAR100':
        training_size_total = 50000

    training_sizes = [50000]
    test_size = 10000
    batch_size = 400
    init_rate = 0.05
    gamma_learning = 0.8
    step_size = 5

    step_size = 10
    gamma = 0.7
    total_epochs = 50
    decay_normal = 0

    epoch_big = 3
    
    hidden_ratio = np.linspace(0.1,2.0,num=10)
    
    if args.dataset== 'MNIST':
        hidden_ratio = [0.04,0.06,0.08,0.1,0.15,0.2,0.25,0.3,2.0]
                
#     if num_train_labels == 2:
#         hidden_ratio = [0.04,0.06,0.08,0.1,0.15,0.2,0.25,2.0]
#         print('mag')


    # total_classes = np.arange(1,100,step=5)
#     training_sizes = [100,200,400,800,1600,3200,6400,12800,25600,50000]
#     training_sizes = [100,200]

    # training_sizes = [10000]
    # gradual_noise = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    diff_gradual  = [] 
    all_normal_losses = [] 
    all_noise_losses = []
    all_ents = [] 
    all_true_bounds = [] 
    all_accuracies = [] 
    all_train_accuracies = [] 
    all_test_losses = [] 


# +
# ------------------------------- Create Dataset and Noisy Version ------------------------------
    transform_train = transforms.Compose(
        [transforms.ToTensor(),
         ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         ])


    if dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform_train)
        dataset_test = torchvision.datasets.MNIST(root='./data', train=False,
                                                download=True, transform=transform_test)
        dataset.data = dataset.data.float().unsqueeze(1)/255.0
        dataset_test.data = dataset_test.data.float().unsqueeze(1)/255.0

    elif dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_test)
        dataset.data = torch.permute(torch.from_numpy(dataset.data),(0,3,1,2))
        print(dataset.data.shape)
        dataset.targets = torch.from_numpy(np.array(dataset.targets))
        dataset.data = dataset.data.float()/255.0
        dataset_test.data = torch.permute(torch.from_numpy(dataset_test.data),(0,3,1,2))
        dataset_test.targets = torch.from_numpy(np.array(dataset_test.targets))
        dataset_test.data = dataset_test.data.float()/255.0
    elif dataset_name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform_train)
        dataset_test = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=transform_test)
        dataset.data = torch.permute(torch.from_numpy(dataset.data),(0,3,1,2))
        dataset.targets = torch.from_numpy(np.array(dataset.targets))
        dataset.data = dataset.data.float()/255.0
        dataset_test.data = torch.permute(torch.from_numpy(dataset_test.data),(0,3,1,2))
        dataset_test.targets = torch.from_numpy(np.array(dataset_test.targets))
        dataset_test.data = dataset_test.data.float()/255.0

#     dataset.data = dataset.data[:training_size_total]
#     dataset.targets = dataset.targets[:training_size_total]
#     print(dataset.data.shape)
    
#     dataset.data = dataset.data.flatten(1,3).unsqueeze(2).unsqueeze(3)

    dataset.data = dataset.data.cuda()  #train_dataset.train_data is a Tensor(input data)
    dataset.targets = dataset.targets.cuda()
    
    
#     dataset_test.data = dataset_test.data.flatten(1,3).unsqueeze(2).unsqueeze(3)

    dataset_test.data = dataset_test.data.cuda()  #train_dataset.train_data is a Tensor(input data)
    dataset_test.targets = dataset_test.targets.cuda()
    
    
    
#     label_max = torch.max(dataset.targets).cpu().numpy()
#     temp_targets = dataset.targets.clone() 
#     # noise_labels = copy(dataset.targets[torch.randperm(torch.numel(dataset.targets))])
#     rand_indices = torch.randperm(torch.numel(dataset.targets))
#     num_to_change = int(labelnoise*float(torch.numel(dataset.targets)))
#     to_change = rand_indices[0:num_to_change]
    
#     for i in range(len(to_change)):
#         dataset.targets[to_change[i]] = rand_another(dataset.targets[to_change[i]], label_max)
        
#     check = dataset.targets==temp_targets
    
#     print(torch.mean(check.float()))
# -

# -----------------------------------------------------------------------------------------

# +
    
    
    
    for temp in range(total_trials): 
        Networks_to_train = [] 
        Networks_to_train_noise = [] 
        Network_to_test = Net_vanilla_cnn_small(input_channels,hidden_ratio[-1],num_train_labels)
        
        for i in range(len(hidden_ratio)):
            Networks_to_train.append(Net_vanilla_cnn_small(input_channels,hidden_ratio[i],num_train_labels))
            Networks_to_train_noise.append(Net_vanilla_cnn_small(input_channels,hidden_ratio[i],num_train_labels))
    
            
        
        # Networks_to_train = [Net_steerinvariant_mnist_scale()]
    
        # listdict = load_dataset(dataset_name, val_splits, training_size)
        # accuracy_all = np.zeros((val_splits,len(Networks_to_train)))
        
        
        dataset2 = deepcopy(dataset)
#         print(dataset2.targets)

        dataset2_test = deepcopy(dataset_test)
        permuted_list = np.random.permutation(label_list)
        label_set = permuted_list[:num_train_labels]
        print('label set:', label_set)
        indices = torch.zeros_like(dataset.targets)
        indices_test = torch.zeros_like(dataset2_test.targets) 


        for tp in range(len(label_set)):
            index_mania = dataset.targets==label_set[tp]
            index_mania = index_mania.float().nonzero()
            index_mania_test =  dataset_test.targets==label_set[tp]
            index_mania_test = index_mania_test.float().nonzero()

            dataset2.targets[index_mania] = tp
            dataset2_test.targets[index_mania_test] = tp
            indices[index_mania]  = 1
            indices_test[index_mania_test] = 1

        dataset2.data = dataset2.data[indices.nonzero(as_tuple=True)]
        dataset2.targets = dataset2.targets[indices.nonzero(as_tuple=True)]
        dataset2_test.data = dataset2_test.data[indices_test.nonzero(as_tuple=True)]
        dataset2_test.targets = dataset2_test.targets[indices_test.nonzero(as_tuple=True)]
        
        
        
        if variable_datasize:
            random_samplesize = int(199 + (len(dataset2.data)-200)*np.random.rand())
            while np.mod(random_samplesize,batch_size)<10:
                random_samplesize = int(199 + (len(dataset2.data)-200)*np.random.rand())
            dataset2.data = dataset2.data[:random_samplesize]
            dataset2.targets = dataset2.targets[:random_samplesize]
        
        dataset2.data = dataset2.data[:random_samplesize]
        dataset2.targets = dataset2.targets[:random_samplesize]
     

        
        ent = estimate_entropy(dataset2.targets,dataset_name)
        
        all_ents.append(ent)
        all_true_bounds.append(ent)
        
        
        
        my_dataset = Dataset(dataset_name, dataset2.data, dataset2.targets)
        my_dataset_test = Dataset(dataset_name, dataset2_test.data, dataset2_test.targets)
        
        


        trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                              shuffle=True,generator=torch.Generator(device='cuda'), num_workers=0)
        testloader = torch.utils.data.DataLoader(my_dataset_test, batch_size=batch_size,
                                              shuffle=False,generator=torch.Generator(device='cuda'), num_workers=0)
        
        
        dataset3 = deepcopy(dataset2)
        
        dataset3.targets = dataset2.targets[torch.randperm(torch.numel(dataset3.targets))]
        
        while torch.abs(torch.mean((dataset3.targets==dataset2.targets).float())-(1.0/num_train_labels))>0.01:
                dataset3.targets = dataset2.targets[torch.randperm(torch.numel(dataset3.targets))]
        print(torch.mean((dataset3.targets==dataset2.targets).float()))
#         for k in range(len(dataset3.targets)):
#             dataset3.targets[k] = rand_another(dataset3.targets[k], num_train_labels-1)
        
        my_dataset2 = Dataset(dataset_name, dataset3.data, dataset3.targets)
        
        trainloader_noise = torch.utils.data.DataLoader(my_dataset2, batch_size=batch_size,
                                              shuffle=True, generator=torch.Generator(device='cuda'),num_workers=0)
        
        diff_array = [] 
        normal_losses = []
        noise_losses = [] 
        
        
        for i in range(len(Networks_to_train)):
#             print(i)
            Networks_to_train[i],normal_loss = train_network_normal(Networks_to_train[i],trainloader, init_rate,total_epochs,step_size,gamma_learning, decay_normal)
            Networks_to_train_noise[i],noise_loss = train_network_noise(Networks_to_train_noise[i],trainloader_noise, init_rate,total_epochs, step_size,gamma_learning, decay_normal)
            diff_array.append(noise_loss-normal_loss)
            normal_losses.append(normal_loss)
            noise_losses.append(noise_loss)
        
        Network_to_test,normal_loss = train_network_normal(Network_to_test,trainloader, init_rate,total_epochs, 50,0.8, decay_normal)
        
#         print(my_dataset.labels)
#         print(my_dataset_test.labsels)
        accuracy,test_loss = test_network(Network_to_test, testloader, my_dataset_test.labels)
        train_accuracy,nouse = test_network(Network_to_test, trainloader, my_dataset.labels)

        print("noise:",noise_losses)
        print("normal:", normal_losses)
        print("diff:", diff_array)
        print(np.max(diff_array))
        print(ent - normal_losses[-1])
        print('Upper bound: ',ent)
        print(accuracy)
        print(train_accuracy-accuracy)
        print('---')
        diff_gradual.append(np.max(diff_array))
        all_normal_losses.append(normal_losses)
        all_noise_losses.append(noise_losses)
        all_accuracies.append(accuracy)
        all_train_accuracies.append(train_accuracy)
        all_test_losses.append(test_loss)

  
    with open('Adam_datasizevariant'+str(variable_datasize)+'_Correlation_test_performance_'+args.dataset +'_' + str(num_train_labels)+'classes_'+ str(total_trials)+'trials_CNN.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([diff_gradual, all_normal_losses, all_noise_losses,all_ents,all_true_bounds,all_accuracies,all_train_accuracies], f)
    
   
