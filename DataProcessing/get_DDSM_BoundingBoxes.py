'''
 This code is for getting all the directions and DICOM file directions that exist in the DDSM folder.
 After getting those, this code is capable of extracting necessary information to csv_files.
 Such info can be the coordinates of a bounding box surrounding a lession or the BI-RADS classification of a lession
'''
from glob import glob
import SimpleITK
import csv
import numpy
import os
from tqdm import tqdm
import pandas
import re
## Get paths of all 'ROI mask images'
mask_paths = glob("/path-to-data/DDSM/CBIS-DDSM-All-doiJNLP-zzWs5zfZ/*/*_[1,2,3,4,5,6,7,8,9]/*/*ROI mask images*/")


## A function that returns the bounding box coordinates from an ndarray
# We assume that there is only one tumor in each mask
def get_BoundingBox(mask_array):
    if not isinstance(mask_array,numpy.ndarray):
        raise TypeError(f"expected mask_array to be type of 'numpy.ndarray' but got {type(mask_array)}")
    lesion = numpy.where(mask_array==255)

    return min(lesion[0]),  max(lesion[0]), min(lesion[1]), max(lesion[1])

## Take a bounding box of each mask image in the paths and write it in a csv_file
def save_BoundingBox(csv_file_name, mask_paths):
    with open(csv_file_name,"w") as csv_file:
        writer = csv.writer(csv_file)
        
        #Header
        writer.writerow(["id","img","mask","x0","x1","y0","y1"])
        
        #Data
        for path in tqdm(mask_paths):
            
            ## Take the dicom file with the largest size, because the file with small size is a cropped ROI
            if os.path.exists(path+"1-2.dcm") and \
               os.stat(path+"1-2.dcm").st_size > os.stat(path+"1-1.dcm").st_size:
                fn = "1-2.dcm"
            else:
                fn = "1-1.dcm"
            mask = SimpleITK.ReadImage(path+fn)
            mask_array = SimpleITK.GetArrayFromImage(mask)[0]            
            bbox = get_BoundingBox(mask_array)

            patient_id = re.search(r"P_\d{5}",path).group()
            im_path = path[:len(path)-56]
            # folder = re.search(r"path[]",path)
            
            row = [patient_id,im_path,path+fn] + list(bbox)
            # print(row,"\n*********************WRITING**************************\n")
            writer.writerow(row)

## Get all the X-ray images of DDSM
def get_ImagePaths(mask_paths):
    image_paths = glob("/path-to-data/DDSM/*/*/*[MLO,CC]/")
    with open("imgs.csv","w") as f:
        writer = csv.writer(f)
        for path in image_paths:
            writer.writerow([path+"1-1.dcm"])

def get_BiRads():
    bboxes = pandas.read_csv("path-to-data/DDSM/csv_Files/DDSM_BoundingBoxes.csv")
    with open("f.csv","w") as f:
        writer = csv.writer(f)
        for path in bboxes["fn"]:
            writer.writerow([re.search(r"P_\d{5}",path).group()])
        




