from shutil import copyfile
import argparse
import os.path
from m_util import *
opt = argparse.ArgumentParser().parse_args()


#######################################################################################################

####



 
### Generates an list of arrays. The values in the arrays represent index values for images. 
### The number of 
def gen_folds(NIM,n):
    idx = np.random.permutation(NIM)
    return [idx[i::n] for i in range(n)]



#### This function will split up the training data into separate folders which can be used for validation



def train_splitter(root_dir, n_folds, class_n):
    
    opt.im_fold=os.path.join(root_dir,class_n)
    opt.split=os.path.join(root_dir,'split',class_n)
    opt.new_fold=os.path.join(root_dir,'fold_')

    sdmkdir(opt.split)

    imlist = []
    test_ratio = 0.25

    for root,_,fnames in sorted(os.walk(opt.im_fold)):
        for fname in fnames:
            if fname.lower().endswith('.png'):
                imlist.append(fname)
                
    nim = len(imlist)
    print("all: ",nim)
    print(imlist)
    list_to_file(opt.split+'/all.txt',imlist)
    nim_test = int(float(nim)*test_ratio)
    print(nim_test)

    for split_idx in range(1):
        test_list= []
        train_list = []

        folds = gen_folds(nim,n_folds)
        print(folds)
        for f in range(n_folds):
            test_list = [imlist[i] for i in folds[f]] 
            train_list = [i for i in imlist if  i not in test_list]
            sdmkdir(opt.new_fold+str(f)+'/train/'+class_n)
            sdmkdir(opt.new_fold+str(f)+'/val/'+class_n)
            for file in train_list:
                copyfile(opt.im_fold+'/'+file,opt.new_fold+str(f)+'/train/'+class_n+'/'+file)
            for file in test_list:
                copyfile(opt.im_fold+'/'+file,opt.new_fold+str(f)+'/val/'+class_n+'/'+file)
            list_to_file(opt.split+"/val_"+str(f)+'.txt',test_list)
            list_to_file(opt.split+"/train_"+str(f)+'.txt',train_list)



## Set number of folds
n_folds=4

class_n='Whales'

root_dir = '/home/ghumphries/Projects/whale/Data'

train_splitter(root_dir,n_folds,'Whales')

train_splitter(root_dir,n_folds,'Water')


#class_n='Water'





        
    
