import os
from threading import main_thread
import numpy
import re
import pandas
from tqdm import tqdm
from density_calculations import calc_density, calc_density_modified, calc_density_with_bins, calc_density_with_mean


#function to load the saved heatmaps and apply thresholds
def load_and_process(threshold_inner_benign,
                    threshold_inner_malignant,
                    threshold_outer_benign,
                    threshold_outer_malignant,
                    path="/home/server/other_projects/breast_cancer/DATA_PATH/temp_files"):

    if not isinstance(threshold_inner_benign, float):
        raise TypeError(f"expected threshold_inner_benign to be float, got {type(threshold_inner_benign)}")

    if not (0 < threshold_inner_benign < 1):
        raise ValueError(f"expected threshold_inner_benign to be in range 0 to 1, but got {threshold_inner_benign}")


    if not isinstance(threshold_inner_malignant, float):
        raise TypeError(f"expected threshold_inner_malignant to be float, got {type(threshold_inner_malignant)}")

    if not (0 < threshold_inner_malignant < 1):
        raise ValueError(f"expected threshold_inner_malignant to be in range 0 to 1, but got {threshold_inner_malignant}")

    
    if not isinstance(threshold_outer_benign, float):
        raise TypeError(f"expected threshold_outer_benign to be float, got {type(threshold_outer_benign)}")

    if not (0 < threshold_outer_benign < 1):
        raise ValueError(f"expected threshold_outer_benign to be in range 0 to 1, but got {threshold_outer_benign}")


    if not isinstance(threshold_outer_malignant, float):
        raise TypeError(f"expected threshold_outer_malignant to be float, got {type(threshold_outer_malignant)}")

    if not (0 < threshold_outer_malignant < 1):
        raise ValueError(f"expected threshold_outer_malignant to be in range 0 to 1, but got {threshold_outer_malignant}")

    files = os.listdir(path)
    files.sort(key = lambda x: int(re.match(r"image(\d+).+\.npy", x)[1]))

    N = len(files) // 3

    main_matrix = numpy.zeros((3, 3))

    for idx in tqdm(range(N)):
        
        temp = files[3* idx:3*idx + 3]
        temp.sort(key = lambda x: 0 if "bbox" in x else 1 if "ben" in x else 2)
        
        with open(os.path.join(path, temp[0]), 'rb') as f:

            bbox = numpy.load(f)

        with open(os.path.join(path, temp[1]), 'rb') as f:

            heatmap_benign = numpy.load(f)

        with open(os.path.join(path, temp[2]), 'rb') as f:
            
            heatmap_malignant = numpy.load(f)
        #the re.match used to get the annotations from the file names
        annot = re.match(".+_(.+).npy", temp[1])[1]        

        if annot in ["2", "3"]:
            true_label = 1
        elif annot in ["4", "4a", "4b", "4c", "5", "6"]:
            true_label = 2
        else:
            true_label = 0


        dens_ben, dens_mal = calc_density(heatmap_benign, heatmap_malignant)


        is_benign = dens_ben > threshold_outer_benign
        is_malignant = dens_mal > threshold_outer_malignant


        if is_benign and is_malignant:
            if dens_ben > dens_mal:
                pred_label = 1
            else:
                pred_label = 2

        elif is_benign:
            pred_label = 1
        elif is_malignant:
            pred_label = 2
        else:
            pred_label = 0

        main_matrix[pred_label, true_label] += 1

    specificity = numpy.zeros(3)    
    sensitivity = numpy.zeros(3)


    ##specificity

    specificity[0] = (main_matrix[1, 1] + main_matrix[2, 2])/\
    (main_matrix[1, 0] + main_matrix[2, 0] + main_matrix[1, 1] + main_matrix[2, 2])
    
    specificity[1] = (main_matrix[0, 0] + main_matrix[2, 2])/\
    (main_matrix[0, 1] + main_matrix[2, 1] + main_matrix[0, 0] + main_matrix[2, 2])
    
    specificity[2] = (main_matrix[0, 0] + main_matrix[1, 1])/\
    (main_matrix[0, 2] + main_matrix[1, 2] + main_matrix[0, 0] + main_matrix[1, 1])


    ##sensitivity
    
    sensitivity[0] = main_matrix[0, 0]/(main_matrix[0].sum())
    sensitivity[1] = main_matrix[1, 1]/(main_matrix[1].sum())
    sensitivity[2] = main_matrix[2, 2]/(main_matrix[2].sum())
    
    
    #mean accuracy
    
    corrects = main_matrix[0, 0] + main_matrix[1, 1] + main_matrix[2, 2]
    mean_accuracy = corrects / main_matrix.sum()


    return main_matrix, specificity, sensitivity, mean_accuracy


def load_and_process_with_mean( threshold_benign,
                                threshold_malignant,
                                path="/home/server/other_projects/breast_cancer/DATA_PATH/temp_files"):

    if not isinstance(threshold_benign, float):
        raise TypeError(f"expected threshold_benign to be float, got {type(threshold_benign)}")

    if not (0 < threshold_benign < 1):
        raise ValueError(f"expected threshold_benign to be in range 0 to 1, but got {threshold_benign}")


    if not isinstance(threshold_malignant, float):
        raise TypeError(f"expected threshold_malignant to be float, got {type(threshold_malignant)}")

    if not (0 < threshold_malignant < 1):
        raise ValueError(f"expected threshold_malignant to be in range 0 to 1, but got {threshold_malignant}")


    files = os.listdir(path)
    files.sort(key = lambda x: int(re.match(r"image(\d+).+\.npy", x)[1]))

    N = len(files) // 3

    main_matrix = numpy.zeros((3, 3))

    for idx in tqdm(range(N)):
        
        temp = files[3* idx:3*idx + 3]
        temp.sort(key = lambda x: 0 if "bbox" in x else 1 if "ben" in x else 2)
        
        with open(os.path.join(path, temp[0]), 'rb') as f:

            bbox = numpy.load(f)

        with open(os.path.join(path, temp[1]), 'rb') as f:

            heatmap_benign = numpy.load(f)

        with open(os.path.join(path, temp[2]), 'rb') as f:
            
            heatmap_malignant = numpy.load(f)
        #the re.match used to get the annotations from the file names
        annot = re.match(".+_(.+).npy", temp[1])[1]        

        if annot in ["2", "3"]:
            true_label = 1
        elif annot in ["4", "4a", "4b", "4c", "5", "6"]:
            true_label = 2
        else:
            true_label = 0


        dens_ben, dens_mal = calc_density_with_mean(heatmap_benign, heatmap_malignant,
                                    box_benign=bbox, box_malignant=bbox)


        is_benign = dens_ben > threshold_benign
        is_malignant = dens_mal > threshold_malignant


        if is_benign and is_malignant:
            if dens_ben > dens_mal:
                pred_label = 1
            else:
                pred_label = 2

        elif is_benign:
            pred_label = 1
        elif is_malignant:
            pred_label = 2
        else:
            pred_label = 0

        main_matrix[pred_label, true_label] += 1

    specificity = numpy.zeros(3)    
    sensitivity = numpy.zeros(3)


    ##specificity

    specificity[0] = (main_matrix[1, 1] + main_matrix[2, 2])/\
    (main_matrix[1, 0] + main_matrix[2, 0] + main_matrix[1, 1] + main_matrix[2, 2])
    
    specificity[1] = (main_matrix[0, 0] + main_matrix[2, 2])/\
    (main_matrix[0, 1] + main_matrix[2, 1] + main_matrix[0, 0] + main_matrix[2, 2])
    
    specificity[2] = (main_matrix[0, 0] + main_matrix[1, 1])/\
    (main_matrix[0, 2] + main_matrix[1, 2] + main_matrix[0, 0] + main_matrix[1, 1])


    ##sensitivity
    
    sensitivity[0] = main_matrix[0, 0]/(main_matrix[0].sum())
    sensitivity[1] = main_matrix[1, 1]/(main_matrix[1].sum())
    sensitivity[2] = main_matrix[2, 2]/(main_matrix[2].sum())
    
    
    #mean accuracy
    
    corrects = main_matrix[0, 0] + main_matrix[1, 1] + main_matrix[2, 2]
    mean_accuracy = corrects / main_matrix.sum()


    return main_matrix, specificity, sensitivity, mean_accuracy



#function to load the saved heatmaps and apply several thresholds at once
def load_and_process_modified(threshold_inner_benign,
                              threshold_inner_malignant,
                              threshold_outer_benign,
                              threshold_outer_malignant,
                              save_result=False):



    if not isinstance(threshold_inner_benign, numpy.ndarray):
        raise TypeError(f"expected threshold_inner_benign to be numpy.ndarra, got {type(threshold_inner_benign)}")

    if not (threshold_inner_benign.ndim == 1):
        raise ValueError(f"expected threshold_inner_benign to have shape [N], but got {threshold_inner_benign.shape}")

    if not numpy.all((0 < threshold_inner_benign) & (threshold_inner_benign < 1)):
        raise ValueError(f"expected threshold_inner_benign values to be in range 0 to 1, but got {threshold_inner_benign}")


    if not isinstance(threshold_inner_malignant, numpy.ndarray):
        raise TypeError(f"expected threshold_inner_malignant to be numpy.ndarra, got {type(threshold_inner_malignant)}")

    if not (threshold_inner_malignant.ndim == 1):
        raise ValueError(f"expected threshold_inner_malignant to have shape [N], but got {threshold_inner_malignant.shape}")

    if not numpy.all((0 < threshold_inner_malignant) & (threshold_inner_malignant < 1)):
        raise ValueError(f"expected threshold_inner_malignant values to be in range 0 to 1, but got {threshold_inner_malignant}")


    if not isinstance(threshold_outer_benign, numpy.ndarray):
        raise TypeError(f"expected threshold_outer_benign to be numpy.ndarra, got {type(threshold_outer_benign)}")

    if not (threshold_outer_benign.ndim == 1):
        raise ValueError(f"expected threshold_outer_benign to have shape [N], but got {threshold_outer_benign.shape}")

    if not numpy.all((0 < threshold_outer_benign) & (threshold_outer_benign < 1)):
        raise ValueError(f"expected threshold_outer_benign values to be in range 0 to 1, but got {threshold_outer_benign}")


    if not isinstance(threshold_outer_malignant, numpy.ndarray):
        raise TypeError(f"expected threshold_outer_malignant to be numpy.ndarra, got {type(threshold_outer_malignant)}")

    if not (threshold_outer_malignant.ndim == 1):
        raise ValueError(f"expected threshold_outer_malignant to have shape [N], but got {threshold_outer_malignant.shape}")

    if not numpy.all((0 < threshold_outer_malignant) & (threshold_outer_malignant < 1)):
        raise ValueError(f"expected threshold_outer_malignant values to be in range 0 to 1, but got {threshold_outer_malignant}")


    if not isinstance(save_result, bool):
        raise TypeError(f"expected result type to be bool but got {type(save_result)}")
    
    path = "/home/server/other_projects/breast_cancer/DATA_PATH/temp_files"
    result_path = "/home/server/other_projects/breast_cancer/breast_cancer_gitlab/HeatmapGenerator"

    files = os.listdir(path)
    files.sort(key = lambda x: int(re.match(r"image(\d+).+\.npy", x)[1]))

    N = len(files) // 3

    corrects_benign = numpy.zeros((threshold_inner_benign.shape[0], threshold_outer_benign.shape[0]))
    corrects_malignant = numpy.zeros((threshold_inner_malignant.shape[0], threshold_outer_malignant.shape[0]))

    total_benign = numpy.zeros((threshold_inner_benign.shape[0], threshold_outer_benign.shape[0]))
    total_malignant = numpy.zeros((threshold_inner_malignant.shape[0], threshold_outer_malignant.shape[0]))

    
    for idx in range(N):

        with open(os.path.join(path, files[3 * idx]), 'rb') as f:

            bbox = numpy.load(f)

        with open(os.path.join(path, files[3 * idx + 1]), 'rb') as f:

            heatmap_benign = numpy.load(f)

        with open(os.path.join(path, files[3 * idx + 2]), 'rb') as f:
            
            heatmap_malignant = numpy.load(f)

        #the re.match used to get the annotations from the file names
        annot = re.match(".+_(.+).npy", files[3 * idx + 1])[1]
        


        dens_ben, dens_mal = calc_density_modified(heatmap_benign, heatmap_malignant,
                                    box_benign=bbox, box_malignant=bbox,
                                    threshold_benign=threshold_inner_benign, 
                                    threshold_malignant=threshold_inner_malignant)


        is_benign = dens_ben[..., None] > threshold_outer_benign
        is_malignant = dens_mal[..., None] > threshold_outer_malignant

        true_label_ben = (annot in ["2", "3"] )
        true_label_mal = (annot in ["4a", "4b", "4c", "4", "5", "6"])


        if true_label_ben is True:
            pred_ben = is_benign
        else:
            pred_ben = False * is_benign

        if true_label_mal is True:
            pred_mal = is_malignant
        else:
            pred_mal = False * is_malignant


        corrects_benign += pred_ben
        corrects_malignant += pred_mal

        if true_label_ben:
            total_benign += 1
        
        if true_label_mal:
            total_malignant += 1        
        
        print(f"image{idx} in process...")

    acc_ben = corrects_benign/total_benign
    print("Benign accuracy for given thresholds:")
    print(acc_ben)


    acc_mal = corrects_malignant/total_malignant
    print("Malignant accuracy for given thresholds:")
    print(acc_mal)

    if save_result:
            
        data_ben = pandas.DataFrame(data=acc_ben, index=[f"{numpy.round(item, 3)}" for item in threshold_inner_benign], columns=[f"{numpy.round(item, 3)}" for item in threshold_outer_benign])
        data_mal = pandas.DataFrame(data=acc_mal, index=[f"{numpy.round(item, 3)}" for item in threshold_inner_malignant], columns=[f"{numpy.round(item, 3)}" for item in threshold_outer_malignant])

        data_ben.to_csv(os.path.join(result_path, "benign.csv"))
        data_mal.to_csv(os.path.join(result_path, "malignant.csv"))


#function to lead the saved heatmaps and apply several thresholds at once using bins
def load_and_process_with_bins(threshold_benign,
                               threshold_malignant,
                               bins,
                               path="/home/server/other_projects/breast_cancer/DATA_PATH/temp_files"):
    


    if not isinstance(bins, numpy.ndarray):
        raise TypeError(f"expected bins to be type of numpy.ndarray, but got {type(bins)}")

    if not (bins.ndim == 1):
        raise ValueError(f"expected bins to have shape [N], but got {bins.shape}")

    if bins.shape[0] < 3:
        raise ValueError(f"expected bins to have at least 3 elements, but got {bins.shape}")

    if not numpy.all((bins >=0) & (bins <=1)):
        raise ValueError(f"expected bins elemets to be in [0, 1]")

    if not numpy.all(bins[1:] > bins[:-1]):
        raise ValueError(f"expected bins to be strictly monotonic sequence")


    files = os.listdir(path)
    files.sort(key = lambda x: int(re.match(r"image(\d+).+\.npy", x)[1]))

    N = len(files) // 3

    main_matrix = numpy.zeros((3, 3))



    for idx in tqdm(range(N)):
        
        temp = files[3* idx:3*idx + 3]
        temp.sort(key = lambda x: 0 if "bbox" in x else 1 if "ben" in x else 2)
        
        with open(os.path.join(path, temp[0]), 'rb') as f:

            bbox = numpy.load(f)

        with open(os.path.join(path, temp[1]), 'rb') as f:

            heatmap_benign = numpy.load(f)

        with open(os.path.join(path, temp[2]), 'rb') as f:
            
            heatmap_malignant = numpy.load(f)
        #the re.match used to get the annotations from the file names
        annot = re.match(".+_(.+).npy", temp[1])[1]
        
        dens_ben, dens_mal = calc_density_with_bins(heatmap_benign, heatmap_malignant, bbox, bins)
        
        true_label = 1 if annot in ["2", "3"] else 2 if annot in ["4a", "4b", "4c", "4", "5", "6"] else 0

        is_benign = dens_ben > threshold_benign
        is_malignant = dens_mal > threshold_malignant


        if is_benign and is_malignant:
            if dens_ben > dens_mal:
                pred_label = 1
            else:
                pred_label = 2

        elif is_benign:
            pred_label = 1
        elif is_malignant:
            pred_label = 2
        else:
            pred_label = 0

        main_matrix[pred_label, true_label] += 1

    specificity = numpy.zeros(3)    
    sensitivity = numpy.zeros(3)


    ##specificity

    specificity[0] = (main_matrix[1, 1] + main_matrix[2, 2])/\
    (main_matrix[1, 0] + main_matrix[2, 0] + main_matrix[1, 1] + main_matrix[2, 2])
    
    specificity[1] = (main_matrix[0, 0] + main_matrix[2, 2])/\
    (main_matrix[0, 1] + main_matrix[2, 1] + main_matrix[0, 0] + main_matrix[2, 2])
    
    specificity[2] = (main_matrix[0, 0] + main_matrix[1, 1])/\
    (main_matrix[0, 2] + main_matrix[1, 2] + main_matrix[0, 0] + main_matrix[1, 1])


    ##sensitivity
    
    sensitivity[0] = main_matrix[0, 0]/(main_matrix[0].sum())
    sensitivity[1] = main_matrix[1, 1]/(main_matrix[1].sum())
    sensitivity[2] = main_matrix[2, 2]/(main_matrix[2].sum())
    
    
    #mean accuracy
    
    corrects = main_matrix[0, 0] + main_matrix[1, 1] + main_matrix[2, 2]
    mean_accuracy = corrects / main_matrix.sum()


    return main_matrix, specificity, sensitivity, mean_accuracy

    