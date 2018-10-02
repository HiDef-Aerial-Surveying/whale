#####################################################################################
##### Grant Humphries
##### September 26, 2018
##### This script will generate training data from full images using shape files
##### to split out where the animals are in space
##### To install GDAL on a virtual environment, follow instructions from @prinsherbert at https://gist.github.com/cspanring/5680334
######################################################################################


import gdal
from m_util import *
import glob
from PIL import Image
import shapefile
import os
import argparse
parser = argparse.ArgumentParser()


### Set argument for terminal use:  python pull_out_target.py --targetdir /home/ghumphries/Projects/whale/Data/fulldata/train/Whales
parser.add_argument('--targetdir')
opt = parser.parse_args()



### Old directory usage for testing
#targetdir = '/home/ghumphries/Projects/whale/Data/fulldata/train/Whales/'


### Create the directory inside of the target directory where cropped images will be stored
outdir = os.path.join(opt.targetdir,'cutout')
sdmkdir(outdir)


### This is to allow gdal to enable exceptions if an error arises
gdal.UseExceptions()

### We use glob.glob to get a list of images with the png extension and full path names
imgs = glob.glob(os.path.join(opt.targetdir,'*.png'))

if not imgs:
    raise ValueError('Your image list is empty - you should check your directory name')

### Looping through the full path names of the images
print('processing images...')
print(os.path.join(opt.targetdir,'*.png'))
for i in imgs:
    print('processing,', i)
    ## All images have a shapefile with the same name (minus the extension)
    shpname = i[:-4]+'.shp'

    ## Open the PNG image and read in the shape file
    ds = gdal.Open(i)
    shp = shapefile.Reader(shpname)
    shapes = shp.shapes()
    
    ## Get lists of the vertices to find where the min and max values are which allows us to create a bounding box to clip
    for shape in shapes:
        xlist = [vertex[0] for vertex in shape.points]
        ylist = [vertex[1] for vertex in shape.points]
    
    minx = min(xlist)
    maxx = max(xlist)
    miny = min(ylist)
    maxy = max(ylist)
    
    
    ## Clips the raster image to the bounding box
    ds = gdal.Translate('new.png',ds,projWin=[minx,abs(maxy),maxx,abs(miny)])
    

    ## bands have to be extracted first before being merged back together to be displayed (going from gdal format to an image)
    band = ds.GetRasterBand(1)
    band2 = ds.GetRasterBand(2)
    band3 = ds.GetRasterBand(3)
    
    img_array1 = band.ReadAsArray()
    img_array2 = band2.ReadAsArray()
    img_array3 = band3.ReadAsArray()
    
    im1 = Image.fromarray(img_array1)
    im2 = Image.fromarray(img_array2)
    im3 = Image.fromarray(img_array3)
    
    ### PIL library allows us to merge multiple bands
    merged = Image.merge("RGB",(im1,im2,im3))
    
    ### Now we use the file name (taking away the whole path name and just using the file name with the .png extension) to save to a new directory
    ### This makes sure that the file names are consistent 

    outname = os.path.join(outdir,i[i.rfind('/')+1:])
    
    ### Save the output
    merged.save(outname)