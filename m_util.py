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
                transforms.RandomRotation(20),
                transforms.Resize(360),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomGrayscale(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(360),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(360),
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


    def test_im(self,device,model_ft,class_names,test_transforms,im):
        A_img = Image.open(im)
        A_img = A_img.resize((224, 224),Image.NEAREST)
        A_img = test_transforms(A_img)
        A_img = torch.unsqueeze(A_img,0)
        A_img = A_img.to(device)
        pred = model_ft(A_img)
        print(pred.max())



    def test_dir(self,device,model_ft,dataloader):
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


    def gen_folds(self,NIM,n):
        idx = np.random.permutation(NIM)
        return [idx[i::n] for i in range(n)]

    
    def create_cv_data(self,root_dir,class_n,n_folds,out_dir):
        dr = os.path.join(root_dir,class_n)
        flist = os.listdir(dr)
        
        ### In this case, we are actually trying to pull out individuals.
        ### There are multiple frames of individuals and this bit of code will create a unique list of all the individual names
        ### This will be used to draw out unique whales to do cross validation
        print('----------------------------------------------------------------------------------')
        print('Identifying individuals animals...')
        nlist = []
        for f in flist:
            tt = f.find('Camera')
            if tt == 0:
                x = f[:8]
            else:
                ind = f.find('_')
                x = f[:ind]
            nlist.append(x)
        imlist = list(set(nlist))

        nim = len(imlist)

        ### Generate random arrays of indices equal to the number of folds - these are indices from the image name list to be drawn out 
        folds = s.gen_folds(nim,n_folds)

        test_list = []
        train_list = []

        ## This part of the code will go through the folds array and extract the tiled images that refer to individual whales
        print('----------------------------------------------------------------------------------')
        for f in range(n_folds):
            print('working on fold', f)
            ### Creates the training and validation directories
            s.sdmkdir(os.path.join(out_dir,'fold_'+str(f),'train',class_n))    
            s.sdmkdir(os.path.join(out_dir,'fold_'+str(f),'val',class_n))

            print('-----------------')
            ### generates lists of unique names 
            test_list_nms = [imlist[i] for i in folds[f]] 
            train_list_nms = [i for i in imlist if  i not in test_list_nms]
            
            testlist = []
            ### First appends a '_' to the end of the unique name, then looks for all files with the name nn and appends to the list
            ### Also done below for the training list        
            for g in test_list_nms:
                nn = g+'_'
                fls = [y for y in flist if y.find(nn) != -1]#         
                testlist = testlist + fls

            trainlist = []     
            for h in train_list_nms:
                nn = h+'_'
                fls = [y for y in flist if y.find(nn) != -1]#         
                trainlist = trainlist + fls
            
            print('Copying files to the training folder', os.path.join(out_dir,'fold_'+str(f),'train',class_n))
            
            ### Takes the files from trainlist and moves them to either the training or validation folders
            for file in trainlist:
                copyfile(os.path.join(dr,file),  os.path.join(out_dir,'fold_'+str(f),'train',class_n,file))

            print('Copying files to the validation folder', os.path.join(out_dir,'fold_'+str(f),'val',class_n))
            for file in testlist:
                copyfile(os.path.join(dr,file),  os.path.join(out_dir,'fold_'+str(f),'val',class_n,file))