#######################################################################################################
#### Script for testing a pytorch, convolutional neural net, using the pre-trained resnet18 model  ####
#### Authors:  Hieu Le & Grant Humphries
#### Date: August 2018
#### This script was written for the Spacewhale project 
#######################################################################################################
#### Usage examples (Linux)
####
####  $python testing_script.py --data_dir /home/ghumphries/spacewhale/test --model MODEL1 --epoch 24
####
#######################################################################################################
#### Setup information
####    To run this script, ensure that you have folders named exactly the same as those in the training data folder
####    For example: 
####    ./test/Water 
####    ./test/Whale
####    IMPORTANT:
####        The images that you want to test should all live in the target folder. For example, if you only want to test for
####        water, then place all the images in the ./test/Water folder. If you want to test for whales, place all the images in 
####        the ./test/Whale folder
####        The data_dir argument should point to the directory ABOVE the training folders.
####        For example, if your directory is:  /home/user/spacewhale/testingdata/Water
####        then --data_dir /home/user/spacewhale/testingdata
#######################################################################################################
### Library imports

from __future__ import print_function, division

from PIL import Image
from PIL import ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from m_util import *
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse

######################################
### Load spacewhale class
s = spacewhale()
##########################################################################
parse = argparse.ArgumentParser()
parse.add_argument('--data_dir')
parse.add_argument('--model')
parse.add_argument('--epoch',type=int,default=24)
parse.add_argument('--class_list', nargs='+',required=True) ## list of classes you want to view at the output
opt = parse.parse_args()
##########################################################################
### Get data from command line arguments
epoch_to_use = 'epoch_'+str(opt.epoch)+'.pth' 
trained_model = './trained_model/'+opt.model+'/'+epoch_to_use
data_dir = opt.data_dir
###########################################################################

## Load the test transforms from the spacewhale class
test_transforms = s.data_transforms['test']

### Load the GPU device if available, or load into a CPU if not
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

### Load up the resnet model, replace the last layer and then load up the pre-trained model 
model_ft = torchvision.models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load(trained_model))
model_ft.eval()


### This loads up all the images in the folder we designated and then applies the transforms from the data_transforms object
image_datasets = datasets.ImageFolder(data_dir, s.data_transforms['test'])
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=10,shuffle=False, num_workers=16)


### We do this here in case we need it - but basically it just creates a dictionary of the classes 
### So that we know what label goes with what class
class_names = image_datasets.classes
keylist = [x for x in range(len(class_names))]
d = {key: value for (key, value) in zip(keylist,class_names)}


### Run the script to test all pictures in a test directory
s.test_dir(device,model_ft,dataloaders)

#################################################################################################################################
#### The below script is for testing one image at a time. 

#data_dir = '/home/ghumphries/Projects/whale/Data/fulldata/test/single_whale/Water'
#fils = os.listdir(data_dir)
#for f in fils:
#    print(f)
#    im = os.path.join(data_dir,f)
#    s.test_im(device,model_ft,class_names,test_transforms,im)



