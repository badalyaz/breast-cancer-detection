import torch
import numpy
import pandas
import SimpleITK
import os
from glob import glob
import pickle

from skimage.transform import resize
from skimage.color import rgb2gray

#Standardizes an image in-place 
def standard_normalize_single_image(image):
    
    image -= numpy.mean(image)
    image /= numpy.maximum(numpy.std(image), 10**(-5))


##This function is used for rescaling INbreast low resolution images
def interpolate(image, scale_factor):

    temp = resize(image, scale_factor,)

    return rgb2gray(temp)[None, ...]


class INbreastLoader(torch.utils.data.Dataset):
    
    def __init__(self,
                 root="/home/server/other_projects/breast_cancer/DATA_PATH/Data/InBreast_from_kaggle",
                 device="cpu",
                 return_torch=False,
                 bbox_filename = "BoundingBoxes.csv",
                 csv_annot_filename = "INbreast.csv",
                 all_img_location = "ALL-IMGS"
                 ):

        if not os.path.exists(root):  
            raise NotADirectoryError()
        
        if not isinstance(return_torch, bool):
            raise TypeError(f"expected return_torch to be bool type, but got {type(return_torch)}")
        
        if not (device in ["cpu","cuda"]):
            raise ValueError(f"expected device to be 'cpu' or 'cuda', but got {device}")

        if (device == "cuda") and (return_torch is False):
            raise PermissionError("not allowed to convert numpy ndarray to gpu!")

        if not os.path.exists(os.path.join(root,bbox_filename)):
            raise FileNotFoundError()

        if not os.path.exists(os.path.join(root, csv_annot_filename)):
            raise FileNotFoundError()

        if not os.path.exists(os.path.join(root, all_img_location)):
            raise NotADirectoryError()

        self.return_torch = return_torch

        self.root = root
        self.all_img_location = all_img_location
        #Read files
        bbox_file = pandas.read_csv(os.path.join(root, bbox_filename))
        annot_file = pandas.read_csv(os.path.join(root, csv_annot_filename))


        self.filenames = list(bbox_file["fn"])
        self.extensions = list(bbox_file["ext"])
        ## Get the coordinates of boxes in (x0,y0,x1,y1) format: In the csv file the format is (x0,x1,y0,y1)
        boxes = numpy.concatenate(bbox_file.iloc[:,[1,3,2,4]].to_numpy()).reshape(-1,4)

        if numpy.any(boxes < 0):
            raise ValueError("negative bbox coordinates")

        self.boxes = boxes

        #Getting BiRads annotations.
        # indices = list(bbox_file["fn"].apply(lambda x: x.split("_")[0]).astype(int))
        indices = list(bbox_file["fn"].apply(lambda x: x.split("_")[0]).astype(str))
        self.birads_annots = list(annot_file.set_index(annot_file["File Name"]).loc[indices]["Bi-Rads"])

        self.device = device

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        
        # This value is the greatest side between the minimum of each side in INbreast dataset that we have 
        min_size = 2560


        if not isinstance(index, int):
            raise TypeError(f"expected index to be int type, but got {type(index)}")

        if not ( 0 <= index < self.__len__()):
            raise IndexError(f"index {index} out of range")

        image = SimpleITK.ReadImage(os.path.join(self.root, self.all_img_location, self.filenames[index]+ self.extensions[index]))
        image_array = SimpleITK.GetArrayFromImage(image).astype(numpy.float)

        bbox = self.boxes[index]

        #Because there is two types of INbreast  images (.dcm, .png)
        #.dcm images have shape (1, H, W)
        #.png images have shape (H, W, 3)

        if image_array.shape[0] == 1:
            h, w = image_array.shape[1:]
        else:
            h, w = image_array.shape[:-1]
        

        scale_factor = numpy.array([w, h, w, h])
        bbox = bbox / scale_factor
        annot = self.birads_annots[index]


        if not numpy.all((bbox >=0) & (bbox <= 1)):
            raise ValueError(f"incorrect box coordinates, out of the range of image shape")


        if numpy.max(image_array.shape) < min_size:
            image_array = interpolate(image_array, 
                            ( 
                              int(image_array.shape[0] * (min_size/image_array.shape[0])),
                              int(image_array.shape[1] * (min_size/image_array.shape[0]))
                            )
                        )        

        standard_normalize_single_image(image_array)

        if (self.return_torch):
            image_array = torch.from_numpy(image_array.astype(numpy.float))
            image_array = image_array.to(self.device)

            bbox = torch.tensor(bbox, dtype=torch.float).to(self.device)
            #scale_factor = torch.tensor(scale_factor, dtype=torch.float).to(self.device)

        return bbox, annot, image_array

class DDSMLoader(torch.utils.data.Dataset):

    def __init__(self, 
                device="cpu",
                return_torch=False):


        if not isinstance(return_torch, bool):
            raise TypeError(f"expected return_torch to be bool type, but got {type(return_torch)}")
        
        if not (device in ["cpu","cuda"]):
            raise ValueError(f"expected device to be 'cpu' or 'cuda', but got {device}")

        if (device == "cuda") and (return_torch is False):
            raise PermissionError("not allowed to convert numpy ndarray to gpu!")

        self.device = device
        self.return_torch = return_torch

        ##csv file where .dcm file paths and annontations are
        csv_path = "/home/server/other_projects/breast_cancer/DATA_PATH/Data/DDSM/csv_Files/DDSM_BoundingBoxes.csv"

        data = pandas.read_csv(csv_path)

        self.filenames = list(data["img"])

        bboxes = numpy.concatenate(data.iloc[:,[-2,-4,-1,-3]].to_numpy()).reshape(-1,4)
        self.bboxes = bboxes

        #birads and (malignant or benign) annontations separately
        self.birads_annots = list(data["assessment"])
        self.pathologies = list(data["pathology"])



    def __len__(self):
        return len(self.bboxes)
    
    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError(f"expected index to be int type, but got {type(index)}")

        if index >= self.__len__():
            raise IndexError(f"index {index} out of range")

        ##Allowing negative indices
        if index < -self.__len__():
            raise IndexError(f"index {index} out of range")
        
        if index < 0:
            index = self.__len__() + index

        #To get full image path, because there are also two sub-folders 
        img_path = glob(self.filenames[index] + "/*/*/*")[0]
        image = SimpleITK.ReadImage(img_path)
        image_array = SimpleITK.GetArrayFromImage(image).astype(numpy.float)


        bbox = self.bboxes[index]
        h, w = image_array.shape[1:]

        ##transforming bbox coordinates in range [0, 1]
        scale_factor = numpy.array([w, h, w, h])
        bbox = bbox / scale_factor
        birads_annot = self.birads_annots[index]
        pathology = self.pathologies[index]


        if not numpy.all((bbox >=0) & (bbox <= 1)):
            raise ValueError(f"incorrect box coordinates, out of the range of image shape")


        #Normalizing images
        standard_normalize_single_image(image_array)

        if (self.return_torch):
            image_array = torch.from_numpy(image_array.astype(numpy.float))
            image_array = image_array.to(self.device)

            bbox = torch.tensor(bbox, dtype=torch.float).to(self.device)
           # scale_factor = torch.tensor(scale_factor, dtype=torch.float).to(self.device)

            return bbox, birads_annot, pathology, image_array

        return bbox, birads_annot, pathology, image_array



class SFULoader(torch.utils.data.Dataset):
    
    def __init__(self, 
                device="cpu",
                return_torch=False):


        if not isinstance(return_torch, bool):
            raise TypeError(f"expected return_torch to be bool type, but got {type(return_torch)}")
        
        if not (device in ["cpu","cuda"]):
            raise ValueError(f"expected device to be 'cpu' or 'cuda', but got {device}")

        if (device == "cuda") and (return_torch is False):
            raise PermissionError("not allowed to convert numpy ndarray to gpu!")

        self.device = device
        self.return_torch = return_torch

        ##pickle file where png file paths and annontations are
        pickle_file = '/home/server/other_projects/breast_cancer/DATA_PATH/Data/SFUniversity/Annotations.pickle'
        with open(pickle_file, 'rb') as handle:
            self.data = pickle.load(handle)
        self.filenames = list(self.data.keys())


    def __len__(self):
        return len(self.data.keys())
    
    def __getitem__(self, index):
        '''
            Structure of the annotations:
                {
                    'image_path': '/case3189/PNGFiles/B_3189_1.RIGHT_MLO.png',
                    'PATIENT_AGE': 65.0,
                    'FILM_TYPE': 'REGULAR',
                    'DENSITY': 2.0,
                    'DIGITIZER': ['LUMISYS', 'LASER'],
                    'LINES': 4616.0,
                    'PIXELS_PER_LINE': 3112.0,
                    'BITS_PER_PIXEL': 12.0,
                    'RESOLUTION': 50.0,
                    'Has_Overlay': True,
                    'TOTAL_ABNORMALITIES': 1,
                    'ABNORMALITIES': {
                        'ABNORMALITY_1': {
                            'LESION_TYPE': 'CALCIFICATION',
                            'TYPE': 'ROUND_AND_REGULAR',
                            'DISTRIBUTION': 'N/A',
                            'ASSESSMENT': 2,
                            'SUBTLETY': 4,
                            'PATHOLOGY': 'BENIGN_WITHOUT_CALLBACK',
                            'TOTAL_OUTLINES': 1,
                            'masks': [
                                {
                                    'mask_x': array([2256, 2255, ...,  2255]),
                                    'mask_y': array([2928, 2929, ...,  2928]),
                                    'bbox': [2191, 2888, 2263, 2944]
                                },

                                {
                                    'mask_x': array([2256, 2255, ...,  2255]),
                                    'mask_y': array([2928, 2929, ...,  2928]),
                                    'bbox': [2191, 2888, 2263, 2944]
                                }
                            ]
                        }
                        'ABNORMALITY_2': {
                            'LESION_TYPE': 'CALCIFICATION',
                            'TYPE': 'ROUND_AND_REGULAR',
                            ...
                            ...
                            ...
                            'masks': ...
                        }
                        ...
                        ...
                    }
                }
        '''
        if not isinstance(index, int):
            raise TypeError(f"expected index to be int type, but got {type(index)}")

        if index >= self.__len__():
            raise IndexError(f"index {index} out of range")

        ##Allowing negative indices
        if index < -self.__len__():
            raise IndexError(f"index {index} out of range")
        
        if index < 0:
            index = self.__len__() + index

        #To get full image path, because there are also two sub-folders 
        img_path = self.data[self.filenames[index]]["image_path"]
        image = SimpleITK.ReadImage(img_path)
        image_array = SimpleITK.GetArrayFromImage(image).astype(numpy.float)

        bboxes = []
        masks = {"x":[],"y":[]}
        bi_rads = {"ASSESSMENT":[],"PATHOLOGY":[]}
        if self.data[self.filenames[index]]["Has_Overlay"]:
            for abn in self.data[self.filenames[index]]["ABNORMALITIES"].values():
                for outline_n in abn["masks"]:
                    bboxes.append(outline_n["bbox"])
                    masks["x"].append(outline_n["mask_x"])
                    masks["y"].append(outline_n["mask_y"])
                bi_rads["ASSESSMENT"].append(abn["ASSESSMENT"])
                bi_rads["PATHOLOGY"].append(abn["PATHOLOGY"])
        #Normalizing images
        standard_normalize_single_image(image_array)

        if (self.return_torch):
            image_array = torch.from_numpy(image_array.astype(numpy.float))
            image_array = image_array.to(self.device)

            return bboxes,masks,bi_rads,image_array

        return bboxes,masks,bi_rads,image_array