#######################################################################################
#### Utility codes and  that are called for spacewhale
#### Authors: Hieu Le & Grant Humphries
#### Date: August 2018
#######################################################################################
from __future__ import print_function, division

import os
import numpy as np
from PIL import Image
from scipy import misc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy


class spacewhale:
    def __init__(self):
        ##### These are the data transforms used throughout the code - they are called on in other scripts
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),                
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }


    def sdmkdir(self,d):
        if not os.path.isdir(d):
            os.makedirs(d)


    def savepatch_train(self,png,w,h,step,size,imbasename):

        ni = np.int32(np.floor((w- size)/step) +2)
        nj = np.int32(np.floor((h- size)/step) +2)

        for i in range(0,ni-1):
            for j in range(0,nj-1):
                name = format(i,'03d')+'_'+format(j,'03d')+'.png'
                misc.toimage(png[i*step:i*step+size,j*step:j*step+size,:]).save(imbasename+name)
        for i in range(0,ni-1):
            name = format(i,'03d')+'_'+format(nj-1,'03d')+'.png'
            misc.toimage(png[i*step:i*step+size,h-size:h,:]).save(imbasename+format(i,'03d')+'_'+format(nj-1,'03d')+'.png')


        for j in range(0,nj-1):
            name = format(ni-1,'03d')+'_'+format(j,'03d')+'.png'
            misc.toimage(png[w-size:w,j*step:j*step+size,:]).save(imbasename+format(ni-1,'03d')+'_'+format(j,'03d')+'.png')
        
        misc.toimage(png[w-size:w,h-size:h,:]).save(imbasename+format(ni-1,'03d')+'_'+format(nj-1,'03d')+'.png')




    def train_model(self, opt, device, dataset_sizes, dataloaders, model, criterion, optimizer, scheduler, num_epochs=25):
        
        since = time.time()

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            for phase in ['train']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                    filename = 'epoch_'+str(epoch)+'.pth'
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                running_errors = 0

                tp=0
                tn=0
                fp=0
                fn=0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    running_errors += torch.sum(preds != labels.data)

                    tp += torch.sum(preds[labels.data==0] == 0)
                    fn += torch.sum(preds[labels.data==0] == 1)
                    fp += torch.sum(preds[labels.data==1] == 0)
                    tn += torch.sum(preds[labels.data==1] ==1)


                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                epoch_err = running_errors.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f} Err: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_err))
                torch.save(model.state_dict(),opt.checkpoint+'/'+filename)

                print('TP: {:.4f}  TN: {:.4f}  FP: {:.4f}  FN: {:.4f}'.format(tp, tn, fp, fn))

        time_elapsed = time.time() - since
        print('-----------------------------------------------------------')

        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        
        print('-----------------------------------------------------------')
        
        #print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        #model.load_state_dict(best_model_wts)
        #return model


        def test_im(self,model_ft,class_names,test_transforms,im):
            A_img = Image.open(im)
            A_img = A_img.resize((224, 224),Image.NEAREST)
            A_img = test_transforms(A_img)
            A_img = torch.unsqueeze(A_img,0)
            A_img = A_img.to(device)
            pred = model_ft(A_img)
            label = class_names[pred.argmax()]
            if label == "Whales":
                print('This is a whale')
            else:
                print('This is water')










def list_to_file(f,list):
    file = open(f,"w")
    for i in list:
        file.write(i[:-4]+"\n")
    file.close()

def read_list(f):
    if os.path.isfile(f):
        with open(f, "r") as ins:
            array = []
            for line in ins:
                array.append(line[:-1])  #-1 due to '\n'
            return array
    return []

def convertMbandstoRGB(tif,imname):
    if tif.shape[0] ==1:
        return tif
    if "QB" in imname:
        return tif[(3,2,1),:,:]
    if "WV" in imname:
        if tif.shape[0] ==8:
            return tif[(5,3,2),:,:]
        if tif.shape[0] ==4:
            return tif[(3,2,1),:,:]
    if "IK" in imname:
        return tif[(3,2,1),:,:]

def to_rgb3b(im):
    # as 3a, but we add an extra copy to contiguous 'C' order
    # data
    # ... where is to_rgb3a?
    return np.dstack([im.astype(np.uint8)] * 3).copy(order='C')
def sdsaveim(savetif,name):
    print(savetif.shape)
    if savetif.dtype == np.uint16:
        savetif = savetif.astype(np.float)
        for i in range(0,savetif.shape[2]):
            savetif[:,:,i] =  savetif[:,:,i] / np.max(savetif[:,:,i]) * 255
        savetif = savetif.astype(np.uint8) 
    if savetif.shape[2] == 3:
        Image.fromarray(savetif.astype(np.uint8)).save(name)
    if savetif.shape[2] == 1:
        Image.fromarray(np.squeeze(savetif.astype(np.uint8)),mode='L').save(name)
    #plt.imshow(savetif[1,:],cmap=cm.gray)




def png2patches(png,step,size):
    step = np.int32(step)
    size=  np.int32(size)
    w,h,z = png.shape
    ni = np.int32(np.floor((w- size)/step) +2)

    nj = np.int32(np.floor((h- size)/step) +2)

    patches = np.zeros((ni,nj,size,size,z))
    for i in range(0,ni-1):
        for j in range(0,nj-1):
            patches[i,j,:,:,:] = png[i*step:i*step+size,j*step:j*step+size,:]
    for i in range(0,ni-1):
        patches[i,nj-1,:,:,:] = png[i*step:i*step+size,h-size:h,:]

    for j in range(0,nj-1):
        patches[ni-1,j,:,:,:] = png[w-size:w,j*step:j*step+size,:]
    patches[ni-1,nj-1,:,:,:] = png[w-size:w,h-size:h,:]
    return patches


def patches2png(patch_fold,imname,w,h,step,size):
    imname=imname[:-4]+'#'
    png = np.zeros((w,h))
    ws = np.zeros((w,h))
    
    ni = np.int32(np.floor((w- size)/step) +2)

    nj = np.int32(np.floor((h- size)/step) +2)
    for i in range(0,ni-1):
        for j in range(0,nj-1):
            
            patch = misc.imread(patch_fold + '/' + imname + format(i,'03d')+'_'+format(j,'03d')+'.png',mode='L')
            png[i*step:i*step+size,j*step:j*step+size]=  png[i*step:i*step+size,j*step:j*step+size]+patch 
            ws[i*step:i*step+size,j*step:j*step+size]=  ws[i*step:i*step+size,j*step:j*step+size]+ 1
    for i in range(0,ni-1):
        patch = misc.imread(patch_fold + '/' + imname + format(i,'03d')+'_'+format(nj-1,'03d')+'.png',mode='L')
        png[i*step:i*step+size,h-size:h] =  png[i*step:i*step+size,h-size:h]+ patch
        ws[i*step:i*step+size,h-size:h] =  ws[i*step:i*step+size,h-size:h]+ 1

    for j in range(0,nj-1):
        patch = misc.imread(patch_fold + '/' + imname + format(ni-1,'03d')+'_'+format(j,'03d')+'.png',mode='L')
        png[w-size:w,j*step:j*step+size]= png[w-size:w,j*step:j*step+size]+ patch
        ws [w-size:w,j*step:j*step+size]= ws [w-size:w,j*step:j*step+size]+ 1
   
    patch = misc.imread(patch_fold + '/' + imname + format(ni-1,'03d')+'_'+format(nj-1,'03d')+'.png',mode='L')
    png[w-size:w,h-size:h] = png[w-size:w,h-size:h]+ patch
    ws [w-size:w,h-size:h] = ws [w-size:w,h-size:h]+ 1
    png = np.divide(png,ws)
    
    return png

def patches2png_legacy(patches,w,h,step,size):
    tif = np.zeros((1,w,h))
    ws = np.zeros((1,w,h))
    
    ni = np.int32(np.floor((w- size)/step) +2)

    nj = np.int32(np.floor((h- size)/step) +2)
    for i in range(0,ni-1):
        for j in range(0,nj-1):
            tif[:,i*step:i*step+size,j*step:j*step+size]=  tif[:,i*step:i*step+size,j*step:j*step+size]+ patches[i,j,:,:,:]
            ws[:,i*step:i*step+size,j*step:j*step+size]=  ws[:,i*step:i*step+size,j*step:j*step+size]+ 1
           
    for i in range(0,ni-1):
        tif[:,i*step:i*step+size,h-size:h] =  tif[:,i*step:i*step+size,h-size:h]+ patches[i,nj-1,:,:,:] 
        ws[:,i*step:i*step+size,h-size:h] =  ws[:,i*step:i*step+size,h-size:h]+ 1

    for j in range(0,nj-1):
        tif[:,w-size:w,j*step:j*step+size]= tif[:,w-size:w,j*step:j*step+size]+ patches[ni-1,j,:,:,:]
        ws[:,w-size:w,j*step:j*step+size]= ws[:,w-size:w,j*step:j*step+size]+ 1
   
    tif[:,w-size:w,h-size:h] = tif[:,w-size:w,h-size:h]+ patches[ni-1,nj-1]
    ws[:,w-size:w,h-size:h] = ws[:,w-size:w,h-size:h]+ 1
    
    tif = np.divide(tif,ws)


    return tif
        
def tif2patches(tif,step,size):
    step = np.int32(step)
    size=  np.int32(size)
    z,w,h = tif.shape
    ni = np.int32(np.floor((w- size)/step) +2)

    nj = np.int32(np.floor((h- size)/step) +2)

    patches = np.zeros((ni,nj,z,size,size))
    for i in range(0,ni-1):
        for j in range(0,nj-1):
            patches[i,j,:,:,:] = tif[:,i*step:i*step+size,j*step:j*step+size]
            #print i*step,i*step+size
    for i in range(0,ni-1):
        patches[i,nj-1,:,:,:] = tif[:,i*step:i*step+size,h-size:h]

    for j in range(0,nj-1):
        patches[ni-1,j,:,:,:] = tif[:,w-size:w,j*step:j*step+size]
    patches[ni-1,nj-1,:,:,:] = tif[:,w-size:w,h-size:h]
    return patches

def patches2tif(patches,w,h,step,size):
    tif = np.zeros((1,w,h))
    ws = np.zeros((1,w,h))
    
    ni = np.int32(np.floor((w- size)/step) +2)

    nj = np.int32(np.floor((h- size)/step) +2)
    
    for i in range(0,ni-1):
        for j in range(0,nj-1):
            tif[:,i*step:i*step+size,j*step:j*step+size]=  tif[:,i*step:i*step+size,j*step:j*step+size]+ patches[i,j,:,:,:]
            ws[:,i*step:i*step+size,j*step:j*step+size]=  ws[:,i*step:i*step+size,j*step:j*step+size]+ 1
           
    for i in range(0,ni-1):
        tif[:,i*step:i*step+size,h-size:h] =  tif[:,i*step:i*step+size,h-size:h]+ patches[i,nj-1,:,:,:] 
        ws[:,i*step:i*step+size,h-size:h] =  ws[:,i*step:i*step+size,h-size:h]+ 1

    for j in range(0,nj-1):
        tif[:,w-size:w,j*step:j*step+size]= tif[:,w-size:w,j*step:j*step+size]+ patches[ni-1,j,:,:,:]
        ws[:,w-size:w,j*step:j*step+size]= ws[:,w-size:w,j*step:j*step+size]+ 1
   
    tif[:,w-size:w,h-size:h] = tif[:,w-size:w,h-size:h]+ patches[ni-1,nj-1]
    ws[:,w-size:w,h-size:h] = ws[:,w-size:w,h-size:h]+ 1
    
    tif = np.divide(tif,ws)


    return tif
        
def savepatch_test(png,w,h,step,size,basename):

    ni = np.int32(np.floor((w- size)/step) +2)
    nj = np.int32(np.floor((h- size)/step) +2)

    for i in range(0,ni-1):
        for j in range(0,nj-1):
            misc.toimage(png[i*step:i*step+size,j*step:j*step+size,:]).save(basename+format(i,'03d')+'_'+format(j,'03d')+'.png')
    for i in range(0,ni-1):
#        patches[i,nj-1,:,:,:] = png[:,i*step:i*step+size,h-size:h]
        misc.toimage(png[i*step:i*step+size,h-size:h,:]).save(basename+format(i,'03d')+'_'+format(nj-1,'03d')+'.png')


    for j in range(0,nj-1):
#        patches[ni-1,j,:,:,:] = png[:,w-size:w,j*step:j*step+size]
        misc.toimage(png[w-size:w,j*step:j*step+size,:]).save(basename+format(ni-1,'03d')+'_'+format(j,'03d')+'.png')
    misc.toimage(png[w-size:w,h-size:h,:]).save(basename+format(ni-1,'03d')+'_'+format(nj-1,'03d')+'.png')


def savepatch_test_with_mask(png,mask,w,h,step,size,imbasename,patchbasename):

    ni = np.int32(np.floor((w- size)/step) +2)
    nj = np.int32(np.floor((h- size)/step) +2)

    for i in range(0,ni-1):
        for j in range(0,nj-1):
            name = format(i,'03d')+'_'+format(j,'03d')+'.png'
            m = mask[i*step:i*step+size,j*step:j*step+size]
            misc.toimage(m,mode='L').save(patchbasename+name)
            misc.toimage(png[i*step:i*step+size,j*step:j*step+size,:]).save(imbasename+name)
    for i in range(0,ni-1):
        
        name = format(i,'03d')+'_'+format(nj-1,'03d')+'.png'
        m = mask[i*step:i*step+size,h-size:h]
        misc.toimage(m,mode='L').save(patchbasename+name)
        misc.toimage(png[i*step:i*step+size,h-size:h,:]).save(imbasename+format(i,'03d')+'_'+format(nj-1,'03d')+'.png')


    for j in range(0,nj-1):
        name = format(ni-1,'03d')+'_'+format(j,'03d')+'.png'
        m = mask[w-size:w,j*step:j*step+size]
        misc.toimage(m,mode='L').save(patchbasename+name)
        misc.toimage(png[w-size:w,j*step:j*step+size,:]).save(imbasename+format(ni-1,'03d')+'_'+format(j,'03d')+'.png')
    
    m= mask[w-size:w,h-size:h]
    misc.toimage(m,mode='L').save(patchbasename+format(ni-1,'03d')+'_'+format(nj-1,'03d')+'.png')
    misc.toimage(png[w-size:w,h-size:h,:]).save(imbasename+format(ni-1,'03d')+'_'+format(nj-1,'03d')+'.png')




