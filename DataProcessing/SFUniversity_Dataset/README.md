## Purpose of this utility
In this directory you can find some tools for converting the old LJPEG images of [SFUniversity's dataset](http://www.eg.usf.edu/cvprg/Mammography/Database.html) into png images
This utility is for converting the old LJPEG images to png.

1. The [DDSMUtility](https://github.com/badalyaz/cancer_detection/tree/interns_branch/DataProcessing/SFUniversity_Dataset/LJPEG2PNG_Converter/DDSMUtility) written in matlab is the modern version of an [old code writen by SFUniversity](http://www.eng.usf.edu/cvprg/Mammography/software/heathusf_v1.1.0.html).
2. [get_data.py](https://github.com/badalyaz/cancer_detection/blob/interns_branch/DataProcessing/SFUniversity_Dataset/LJPEG2PNG_Converter/SFUniversity/get_data.py) is a script for converting the LJPEG images in Windows enviroment and passing it to another system automatically.
3. The [check_PNGcount.py](https://github.com/badalyaz/cancer_detection/blob/interns_branch/DataProcessing/SFUniversity_Dataset/check_PNGcount.py) script checks whether all the images were converted to png or not.
4. The [get_SFUniversity_Masks_Boxes.py](https://github.com/badalyaz/cancer_detection/blob/interns_branch/DataProcessing/SFUniversity_Dataset/get_SFUniversity_Masks_Boxes.py) script gets all the annotations from .ics and .OVERLAY files and creates a pickle file with dictionaries. More details can be observed in the script's file. 
   
   

![Examples from the SFU Dataset](https://github.com/badalyaz/cancer_detection/blob/interns_branch/DataProcessing/SFUniversity_Dataset/test.png "Some examples from the SFUniversity dataset")


## Requirements
- [Requirements for DDSMUtility](https://github.com/badalyaz/cancer_detection.git/)
  

## Usage Details
#### Converting the LJPEG images
- To use the DDSMUtility you need to change the paths in the matlab files.
  - There is am Installation [guide](https://github.com/badalyaz/cancer_detection/blob/interns_branch/DataProcessing/SFUniversity_Dataset/LJPEG2PNG_Converter/DDSMUtility/Tutorial.pdf) for the utility. Follow the steps mentioned in it 
  - In [openDDSMLJPEGAndConvertToPNG.m](https://github.com/badalyaz/cancer_detection/blob/interns_branch/DataProcessing/SFUniversity_Dataset/LJPEG2PNG_Converter/DDSMUtility/openDDSMLJPEGAndConvertToPNG.m) change paths to your dataset with LJPEG images `lines **53** and **56**`. The changed directory must contain a folder which will contain other folders with LJPEG images in it. So the code works in the folders with 2 depth. Be careful with "**/**" and "**\\**" signs, write the paths as they are written.
- In order to convert all the data you will need a Windows based machine. As we worked on a Ubuntu system, we created a script which copies the data from server to a Windows based machine, converts it via the DDSMUtility and sends it to the server. [The script](https://github.com/badalyaz/cancer_detection/blob/interns_branch/DataProcessing/SFUniversity_Dataset/LJPEG2PNG_Converter/SFUniversity/get_data.py) does the mentioned actions automatically. For that you just need to:
  - Get the path to each image in the dataset from the host which contains the dataset, create a csv file with them and move that file to the machine which will convert the data.
  - Set the path for the root folder of the dataset `**line 8**`
  - Set the path to csv file, which contains the paths to each image from host `**line 9**`
  - Set the hostname@ip_address `**line 11**`
  - Set the host password `**line 12**`
- Set the path to openDDSMLJPEGAndConvertToPNG.m file in [runMatlab.bat]
- After the convertion is done, you can check if everything was converted with [check_PNGcount.py](https://github.com/badalyaz/cancer_detection/blob/interns_branch/DataProcessing/SFUniversity_Dataset/check_PNGcount.py) script. It will give you the names of the missing files, also if there are any duplicates.

#### Getting the annotations for all the images
- For this you need to just change the paths to the dataset in the get_SFUniversity_Masks_Boxes.py script. `**line247**`


