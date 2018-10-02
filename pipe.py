import torch
import os.path
import argparse
from scipy import misc
from models.models import create_model
from data.png_dataset import PngDataset
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from tif_handle import TIF_H
from m_util import *
import time
import numpy as np
import rasterio


class Pipe:
    def __init__(self,input,output):
        self.import_model()
        self.output = output
        sdmkdir(self.output)
        self.input =  input


    def import_model(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_ft = torchvision.models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        model_ft = model_ft.to(device)
        model_ft.load_state_dict(torch.load('./trained_model/fold_3/epoch_24.pth'))
        model_ft.eval() #set model to eval mode
        self.network = model_ft
    
    def list_tif_predict(self,file):
        root = '/gpfs/projects/LynchGroup/Orthoed/'
        imlist =[]
        imnamelist =[]
        f = open(file,'r')
        while True:
            line = f.readline()
            if not line:break
            imnamelist.append(line.split()[0] )
        print(imnamelist)

        for name in imnamelist :
            self.tif_predict(root+name)
    def dir_tif_predict(self,fold):
        imlist =[]
        imnamelist =[]
        for root,_,fnames in sorted(os.walk(fold)):
            for fname in fnames:
                if fname.endswith('.tif') and 'M1BS' in fname:
                    path = os.path.join(root,fname)
                    imlist.append(path)
                    imnamelist.append(fname)
        print(imnamelist)
        for name in imlist :
            if not name in ['orthoWV02_11FEB191312281-P1BS-10300100098AAB00_u08rfAEAC.tif']:
                try:

                    self.tif_predict(name)
                except:
                    print('failed')
    def tif_predict(self,file):
        try:    
            print(file)
            basename = os.path.basename(file)
            if not os.path.isfile(self.output+'/'+basename):
                tif = TIF_H(file)
                tif.get_png()
                outpng = self.png_predict(tif.png)
                print(outpng) 
                tif.profile.update(dtype=rasterio.uint8, count=1)
                with rasterio.open(self.output+'/'+basename, 'w', **tif.profile) as dst:
                        dst.write(outpng.astype(rasterio.uint8), 1)
        except:
            print("failed")

            
    def png_predict(self,im):
        last = time.time()
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        opt = parser.parse_args()
        opt.step = 92
        opt.size = 256
        w,h,c = im.shape
        patches = png2patches(im,opt.step,opt.size)
        print(patches.shape)
        elapsed_time = time.time() - last
        last = time.time()
        print('im 2 patches: %0.4f'%(elapsed_time))

        orishape = np.asarray(patches.shape)
        orishape[-1] = 1

        patches = np.reshape(patches, (-1,256,256,3))
        outshape  = np.asarray(patches.shape)
        outshape[3] = 1
        patches = np.transpose(patches,(0,3,1,2))
        s = np.asarray(patches.shape)
        s[1] = 1
        bs = 92
        n_patches = patches.shape[0]
        out = np.zeros(s) 
        print('numbers of patches %d'%(n_patches))
        print('Processing all the patches')
        for i in range(0,n_patches,bs):
            batch  = patches[i:i+bs,:,:,:]
        
            #need to process the batch here:
            temp = self.network.(batch)
            #need to convert the raw prediction into an image of opt.size x opt.size 

            #out[i:i+bs,:,:,:] = temp['raw_out']
            
            #print(temp['raw_out'].shape)

        elapsed_time = time.time() - last
        last = time.time()
        print('patches 2 prediction: %0.4f'%(elapsed_time))

        print(patches.shape)
        print(out.shape)
        out = np.reshape(out,outshape)
        out = np.reshape(out,(orishape[0],orishape[1],outshape[3],outshape[1],outshape[2]))

        print(out.shape)
        print('check')
        outpng = patches2png_legacy(out,w,h,opt.step,opt.size)
        outpng = np.transpose(outpng,(1,2,0))
        outpng = np.squeeze(outpng) 
        print(np.amax(outpng))
        print(np.amin(outpng))
        outpng = (outpng + 1)/2
        #outpng[outpng<0.5] = 0
        #outpng[outpng>=0.5] = 1
        outpng = outpng*255
        return outpng



a = Pipe('','./test_results_3/')
a.list_tif_predict('full.txt')

#a.tif_predict('/gpfs/projects/LynchGroup/Orthoed/WV02_20160119013349_1030010050B0C500_16JAN19013349-M1BS-500637522050_01_P001_u08rf3031.tif')
#a.dir_tif_predict('/gpfs/projects/LynchGroup/Penguin_workstation/Train_all/raw/train/')







if pred[0,0] > pred[0,1]:
        print("Whale")
    else:
        print("not whale")