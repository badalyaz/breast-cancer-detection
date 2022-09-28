import imageio
import numpy
import os
import sys


from Custom_functions_and_classes.datasets import INbreastLoader, DDSMLoader
from .data_generation import add_empty_rois


from ..heatmaps.run_producer_single import produce_heatmaps
from ..heatmaps.run_producer import run_producer as run_producer


def save_image_as_png(image, filename):
    """
    Saves image as png files while preserving bit depth of the image
    """
    imageio.imwrite(filename, image)


#function to save the generated heatmaps
def save_heatmaps(model, parameters, device="cuda"):
    
    path = "/home/server/other_projects/breast_cancer/DATA_PATH/temp_files"

    if not os.path.exists(path):

        os.mkdir(path)

    data_0 = INbreastLoader()
    data_1 = DDSMLoader()

    #from INbreast
    for index in range(0, len(data_0)):

        print(f"image{index} in process...")

        heatmap_benign, heatmap_malignant, bbox, annot = produce_heatmaps(data_0, index, parameters, model, device)

        numpy.save(os.path.join(path, f"image{index}_ben_{annot}"), heatmap_benign)
        numpy.save(os.path.join(path, f"image{index}_mal_{annot}"), heatmap_malignant)
        numpy.save(os.path.join(path, f"image{index}_bbox"), bbox)


    bound = len(data_0)
    
    #from DDSM
    for index in range(0, 100):

        print(f"image{bound + index} in process...")

        heatmap_benign, heatmap_malignant, bbox, annot = produce_heatmaps(data_1, index, parameters, model, device)

        numpy.save(os.path.join(path, f"image{index + bound}_ben_{annot}"), heatmap_benign)
        numpy.save(os.path.join(path, f"image{index + bound}_mal_{annot}"), heatmap_malignant)
        numpy.save(os.path.join(path, f"image{index + bound}_bbox"), bbox)
        
    bound += len(101)

    #add rois that corresponds class 0
    add_empty_rois(bound)

