import os
from random import sample
import numpy
import re

def add_empty_rois(bound):
    
    if not isinstance(bound, int):
        raise TypeError(f"expected bound to be int, but got {type(bound)}")

    if bound <= 0:
        raise ValueError(f"wrong bound value {bound}")

    N = 30
    path = "/home/server/other_projects/breast_cancer/DATA_PATH/temp_files"

    files = os.listdir(path)    
    files.sort(key = lambda x: int(re.match(r"image(\d+).+\.npy", x)[1]))

    ##Take N samples from files
    M = range(len(files) // 3)
    to_get = sample(M,N)

    for idx in to_get:
        
        ##Sort to get the required order
        temp = files[3* idx:3*idx + 3]
        temp.sort(key = lambda x: 0 if "bbox" in x else 1 if "ben" in x else 2)


        with open(os.path.join(path, temp[0]), 'rb') as f:

            bbox = numpy.load(f)

        with open(os.path.join(path, temp[1]), 'rb') as f:

            heatmap_benign = numpy.load(f)

        with open(os.path.join(path, temp[2]), 'rb') as f:
            
            heatmap_malignant = numpy.load(f)


        bbox = bbox * numpy.array([heatmap_benign.shape[-1], heatmap_benign.shape[-2], heatmap_benign.shape[-1], heatmap_benign.shape[-2]])

        annot = re.match(".+_(.+).npy", temp[1])[1]

        while True:

            
            w = numpy.random.randint(350, 450)
            h = numpy.random.randint(350, 450)
            
            x_min = numpy.random.randint(0, heatmap_benign.shape[-1] - w)
            y_min = numpy.random.randint(0, heatmap_benign.shape[0] - h)

            x_max = x_min + w
            y_max = y_min + h

            box = numpy.array([x_min, y_min, x_max, y_max])

            if IOU(bbox, box) < 0.2:
                break

        annot = "1"

        box = box / numpy.array([heatmap_benign.shape[-1], heatmap_benign.shape[-2], heatmap_benign.shape[-1], heatmap_benign.shape[-2]])
        print(f"image{idx + bound} in process...")

        numpy.save(os.path.join(path, f"image{bound + idx}_ben_{annot}"), heatmap_benign)
        numpy.save(os.path.join(path, f"image{bound + idx}_mal_{annot}"), heatmap_malignant)
        numpy.save(os.path.join(path, f"image{bound + idx}_bbox"), box)
 
        
def IOU(box1, box2):
	""" We assume that the box follows the format:
		box1 = [x1,y1,x2,y2], and box2 = [x3,y3,x4,y4],
		where (x1,y1) and (x3,y3) represent the top left coordinate,
		and (x2,y2) and (x4,y4) represent the bottom right coordinate """
	x1, y1, x2, y2 = box1	
	x3, y3, x4, y4 = box2
	x_inter1 = max(x1, x3)
	y_inter1 = max(y1, y3)
	x_inter2 = min(x2, x4)
	y_inter2 = min(y2, y4)
	width_inter = abs(x_inter2 - x_inter1)
	height_inter = abs(y_inter2 - y_inter1)
	area_inter = width_inter * height_inter
	width_box1 = abs(x2 - x1)
	height_box1 = abs(y2 - y1)
	width_box2 = abs(x4 - x3)
	height_box2 = abs(y4 - y3)
	area_box1 = width_box1 * height_box1
	area_box2 = width_box2 * height_box2
	area_union = area_box1 + area_box2 - area_inter
	iou = area_inter / area_union
	return iou

