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

s = spacewhale()


parse = argparse.ArgumentParser()
parse.add_argument('--data_dir')
parse.add_argument('--model')
parse.add_argument('--epoch',type=int,default=24)
opt = parse.parse_args()

#trained_model = os.path.join('./trained_model',opt.model,'epoch_24.pth')
#data_dir = opt.data_dir  

epoch_to_use = 'epoch_'+str(opt.epoch)+'.pth' 

trained_model = '/home/ghumphries/Projects/whale/trained_model/TEST4/'+epoch_to_use
data_dir = '/home/ghumphries/Projects/whale/Data/fulldata/test/samples'




test_transforms = s.data_transforms['test']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

model_ft = torchvision.models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)


model_ft.load_state_dict(torch.load(trained_model))
model_ft.eval()

image_datasets = datasets.ImageFolder(data_dir, s.data_transforms['test'])
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=10,shuffle=False, num_workers=16)


class_names = image_datasets.classes
keylist = [x for x in range(len(class_names))]
d = {key: value for (key, value) in zip(keylist,class_names)}



    
def test_dir(model_ft,dataloader):
    tp=0
    fp=0
    tn=0
    fn=0
    for im, labs in dataloader:
            

        im, labs = im.to(device), labs.to(device)        
        outputs = model_ft(im)
     
        outputs = outputs
        _,preds = torch.max(outputs,1)
        tp = tp+ torch.sum(preds[labs==0] == 0)
        fn = fn+ torch.sum(preds[labs==0] == 1)
        fp = fp +torch.sum(preds[labs==1] == 0)
        tn = tn + torch.sum(preds[labs==1] ==1)
        
    print('Correctly Identified as Water: '+ str(float(tp)))
    print('Correctly Identified as Whales: '+ str(float(tn)))
    print('Misidentified as Water: '+ str(float(fp)))    
    print('Misidentified as Whales: '+ str(float(fn)))
                       
    prec = float(tp)/float(tp+fp)
    recall =  float(tp)/ float(tp+fn)
    print("prec: %f, recall: %f"%(prec,recall))



test_dir(model_ft,dataloaders)

im = 'whale.png'
s.test_im(model_ft,class_names,test_transforms,im)
im = 'water.png'
s.test_im(model_ft,class_names,test_transforms,im)


