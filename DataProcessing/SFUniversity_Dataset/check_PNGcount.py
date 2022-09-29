'''
    A simple mechanism for checking whether the PNG files generated are correct or not.
'''

from glob import glob
from os import dup

## Taking the paths to all of the .png and .ljpeg images in the SFU dataset
pngs = glob("/path-to-your-png-images/SFUniversity/*/*/*/*/*/*/*/*/PNGFiles/*")
ljpegs = glob("/path-to-your-LJPEG-images/SFUniversity/*/*/*/*/*/*/*/*/PNGFiles/../*.LJPEG")
pngs = list(set(pngs))


print(f"Comparing {len(pngs)} '.png' files with {len(ljpegs)} '.LJPEG' files")

## Creating empty lists where the missing data, checked data and duplicates will be stored
missing = []
checked = []
duplicates = []


## Comparing all the pathes and filenames
for ljpeg in ljpegs:
    #Extracting the LJPEG filename and its case folder path from its path
    ljpeg_filename = ljpeg[ljpeg.rfind("/")+1:ljpeg.rfind(".")]
    ljpeg_case_folder = ljpeg[:ljpeg.rfind("/PNGFiles")]

    are_same=False
    for png in pngs:
        #Extracting the png filename and its case folder path from its path
        png_filename = png[png.rfind("/")+1:png.rfind(".")]
        png_case_folder = png[:png.rfind("/PNGFiles")]

        #Checking if they are the same
        if ljpeg_filename == png_filename and ljpeg_case_folder==png_case_folder:
            if (png in checked):
                #If the image was already checked, then its a duplicate
                duplicates.append(png)
            else:
                #If not, then mark is as checked
                checked.append(png)
            are_same=True
            break
    #If not the same add value to 'missing' list
    if not are_same:
        missing.append(ljpeg_case_folder+"/"+ljpeg_filename+".LJPEG")

if len(missing) == 0 and len(duplicates)==0:
    print("Everything is ok, all the images are converted")
else:
    print(f"png images for these files are missing --> {missing}\n\n")
    print(f"These files are duplicates --> {duplicates}")