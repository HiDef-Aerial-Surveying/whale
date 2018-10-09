#######################################################################################################
### Code for splitting training and testing data for cross-validation
### Authors: Hieu Le and Grant Humphries
### Date:    Oct, 2018
#######################################################################################################
###  Setup information
###     To run this in command line, the root_dir option is the directory above your whales / water imagery
###     The class_list argument is a list of the folders that you want to use in the cross-validation
###     e.g.,   python split_train_test.py --root_dir ./Data/fulldata/train --n_folds 4 --out_dir ./Data/fulldata/cv --class_list Water Whales
#######################################################################################################
### Load libraries
from shutil import copyfile
import argparse
import os
import os.path
import numpy as np
from m_util import *

parse = argparse.ArgumentParser()
parse.add_argument('--root_dir',default='./Data/fulldata/train')
parse.add_argument('--n_folds',default=4)
parse.add_argument('--out_dir', default='./Data/fulldata/crossvalidation')
parse.add_argument('--class_list', nargs='+',required=True)
opt = parse.parse_args()
#######################################################################################################
## load the spacewhale class
s = spacewhale()
 
#### Load the options from the arguments
n_folds=opt.n_folds
root_dir=opt.root_dir
out_dir=opt.out_dir
classlist=opt.class_list


##### Loop through the files and run the create cross-validation data 
for i in classlist:
    print('-----------------------------------------------------------------------------')
    print('create cross validation data for', i)
    s.create_cv_data(root_dir,i,n_folds,out_dir)

print('-----------------------------------------------------------------------------')
print('COMPLETE')








