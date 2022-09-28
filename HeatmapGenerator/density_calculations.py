import numpy


#this function is used for calculating the densities with given thresholds
def calc_density(heatmap_benign,
                heatmap_malignant,
                box_benign,
                box_malignant,
                threshold_benign=0.3,
                threshold_malignant=0.3
               ):
               
#### this section of conditions is for error-checking ####
    if not isinstance(heatmap_benign, numpy.ndarray):
        raise TypeError(f"expected heatmap_benign to be type of numpy.ndarray, but got {type(heatmap_benign)}")

    if not isinstance(heatmap_malignant, numpy.ndarray):
        raise TypeError(f"expected heatmap_malignant to be type of numpy.ndarray, but got {type(heatmap_malignant)}")

    if heatmap_benign.ndim != 2:
        raise ValueError(f"expected heatmap_benign to have size [N, M], but got {heatmap_benign.shape}")

    if heatmap_malignant.ndim != 2:
        raise ValueError(f"expected heatmap_malignant to have size [N, M], but got {heatmap_benign.shape}")

    if heatmap_malignant.shape != heatmap_benign.shape:
        raise ValueError(f"expected both heatmaps to have the same size, but got {heatmap_benign.shape} and {heatmap_malignant.shape}")

    if not isinstance(threshold_benign, float):
        raise TypeError(f"expected threshold_benign to be float, but got {type(threshold_benign)}")

    if not isinstance(threshold_malignant, float):
        raise TypeError(f"expected threshold_malignant to be float, but got {type(threshold_malignant)}")

    if not (0 < threshold_benign < 1):
        raise ValueError(f"expected threshold_benign to be in range (0, 1), but got {threshold_benign} ")

    if not (0 < threshold_malignant < 1):
        raise ValueError(f"expected threshold_malignant to be in range (0, 1), but got {threshold_malignant} ")

    if numpy.any(box_benign[0:2] >= box_benign[2:4]):
        raise ValueError("redundant box_benign")

    if numpy.any(box_malignant[0:2] >= box_malignant[2:4]):
        raise ValueError("redundant box_malignant")
################################################################

    #get height an weight
    h, w = heatmap_malignant.shape

    #this is used for rescaling the coordinates of boxes
    scale_factor = numpy.array([w, h, w, h], dtype=float)

    #get box coordinates by rescaling the given ones with scale_factor
    b_x_min, b_y_min, b_x_max, b_y_max = (box_benign * scale_factor).astype(int)
    m_x_min, m_y_min, m_x_max, m_y_max = (box_malignant * scale_factor).astype(int)

    #find the positive pixels count according to threshold for malignant and beningn heatmaps
    ben_positive_sum = (heatmap_benign[b_y_min:b_y_max, b_x_min:b_x_max]>threshold_benign).sum()
    mal_positive_sum = (heatmap_malignant[m_y_min:m_y_max, m_x_min:m_x_max]>threshold_malignant).sum()

    #get total number of pixels (bbox_width*bbox_height)
    ben_total = (b_y_max-b_y_min) * (b_x_max-b_x_min)
    mal_total = (m_y_max-m_y_min) * (m_x_max-m_x_min)

    #compute the density of positive cells in heatmaps
    ben_dens = ben_positive_sum / ben_total
    mal_dens = mal_positive_sum / mal_total


    return ben_dens, mal_dens


def calc_density_with_mean(heatmap_benign,
                            heatmap_malignant,
                            box_benign,
                            box_malignant
               ):
               
#### this section of conditions is for error-checking ####
    if not isinstance(heatmap_benign, numpy.ndarray):
        raise TypeError(f"expected heatmap_benign to be type of numpy.ndarray, but got {type(heatmap_benign)}")

    if not isinstance(heatmap_malignant, numpy.ndarray):
        raise TypeError(f"expected heatmap_malignant to be type of numpy.ndarray, but got {type(heatmap_malignant)}")

    if heatmap_benign.ndim != 2:
        raise ValueError(f"expected heatmap_benign to have size [N, M], but got {heatmap_benign.shape}")

    if heatmap_malignant.ndim != 2:
        raise ValueError(f"expected heatmap_malignant to have size [N, M], but got {heatmap_benign.shape}")

    if heatmap_malignant.shape != heatmap_benign.shape:
        raise ValueError(f"expected both heatmaps to have the same size, but got {heatmap_benign.shape} and {heatmap_malignant.shape}")

    if numpy.any(box_benign[0:2] >= box_benign[2:4]):
        raise ValueError("redundant box_benign")

    if numpy.any(box_malignant[0:2] >= box_malignant[2:4]):
        raise ValueError("redundant box_malignant")
################################################################

    #get height an weight
    h, w = heatmap_malignant.shape

    #this is used for rescaling the coordinates of boxes
    scale_factor = numpy.array([w, h, w, h], dtype=float)

    #get box coordinates by rescaling the given ones with scale_factor
    b_x_min, b_y_min, b_x_max, b_y_max = (box_benign * scale_factor).astype(int)
    m_x_min, m_y_min, m_x_max, m_y_max = (box_malignant * scale_factor).astype(int)

    #compute the density of positive cells in heatmaps
    ben_dens = heatmap_benign[b_y_min:b_y_max, b_x_min:b_x_max].mean()
    mal_dens = heatmap_malignant[m_y_min:m_y_max, m_x_min:m_x_max].mean()


    return ben_dens, mal_dens



#this function is used for calculating the densities with given vectors of thresholds in one go
def calc_density_modified(heatmap_benign,
                heatmap_malignant,
                box_benign,
                box_malignant,
                threshold_benign,
                threshold_malignant
               ):

#### this section of conditions is for error-checking ####
    if not isinstance(heatmap_benign, numpy.ndarray):
        raise TypeError(f"expected heatmap_benign to be type of numpy.ndarray, but got {type(heatmap_benign)}")

    if not isinstance(heatmap_malignant, numpy.ndarray):
        raise TypeError(f"expected heatmap_malignant to be type of numpy.ndarray, but got {type(heatmap_malignant)}")

    if heatmap_benign.ndim != 2:
        raise ValueError(f"expected heatmap_benign to have size [N, M], but got {heatmap_benign.shape}")

    if heatmap_malignant.ndim != 2:
        raise ValueError(f"expected heatmap_malignant to have size [N, M], but got {heatmap_benign.shape}")

    if heatmap_malignant.shape != heatmap_benign.shape:
        raise ValueError(f"expected both heatmaps to have the same size, but got {heatmap_benign.shape} and {heatmap_malignant.shape}")

    if not isinstance(threshold_benign, numpy.ndarray):
        raise TypeError(f"expected threshold_benign to be float, but got {type(threshold_benign)}")

    if not isinstance(threshold_malignant, numpy.ndarray):
        raise TypeError(f"expected threshold_malignant to be float, but got {type(threshold_malignant)}")

    if not (threshold_benign.ndim == 1):
        raise ValueError(f"expected threshold_benign to have size [N], but got {threshold_benign.shape}")

    if not (threshold_malignant.ndim == 1):
        raise ValueError(f"expected threshold_malignant to have size [N], but got {threshold_malignant.shape}")

    if not numpy.all((threshold_benign > 0) & (threshold_benign < 1)):
        raise ValueError(f"expected threshold_benign to be in range (0, 1), but got {threshold_benign} ")

    if not numpy.all((threshold_malignant > 0) & (threshold_malignant < 1)):
        raise ValueError(f"expected threshold_malignant to be in range (0, 1), but got {threshold_malignant} ")

    if numpy.any(box_benign[0:2] >= box_benign[2:4]):
        raise ValueError("redundant box_benign")

    if numpy.any(box_malignant[0:2] >= box_malignant[2:4]):
        raise ValueError("redundant box_malignant")
################################################################

    #get height an weight
    h, w = heatmap_malignant.shape

    #this is used for rescaling the coordinates of boxes
    scale_factor = numpy.array([w, h, w, h], dtype=float)

    #get box coordinates by rescaling the given ones with scale_factor
    b_x_min, b_y_min, b_x_max, b_y_max = (box_benign * scale_factor).astype(int)
    m_x_min, m_y_min, m_x_max, m_y_max = (box_malignant * scale_factor).astype(int)

    #find the positive pixels counts according to thresholds for malignant and beningn heatmaps
    #the result is a vector
    ben_positive_sum = numpy.sum(heatmap_benign[b_y_min:b_y_max, b_x_min:b_x_max][..., None]>threshold_benign, axis=(0, 1))
    mal_positive_sum = numpy.sum(heatmap_malignant[m_y_min:m_y_max, m_x_min:m_x_max][..., None]>threshold_malignant, axis=(0, 1))

    #get total number of pixels (bbox_width*bbox_height)
    ben_total = (b_y_max-b_y_min) * (b_x_max-b_x_min)
    mal_total = (m_y_max-m_y_min) * (m_x_max-m_x_min)

    #compute the density of positive cells in heatmaps for each vector
    ben_dens = ben_positive_sum / ben_total
    mal_dens = mal_positive_sum / mal_total

    return ben_dens, mal_dens

#this function is used for calculating the densities with given vectors of thresholds in one go, using bins
def calc_density_with_bins(heatmap_benign,
                            heatmap_malignant,
                            bbox,
                            bins
                        ):
    
    
    h, w = heatmap_benign.shape

    #this is used for rescaling the coordinates of boxes
    scale_factor = numpy.array([w, h, w, h], dtype=float)
    
    x_min, y_min, x_max, y_max = (bbox * scale_factor).astype(int)    
    
    #returns counts of elements in each bin
    counts_ben = numpy.unique(numpy.digitize(heatmap_benign[y_min:y_max, x_min:x_max], bins=bins), return_index=True, return_counts=True)[-1]
    counts_mal = numpy.unique(numpy.digitize(heatmap_malignant[y_min:y_max, x_min:x_max], bins=bins), return_index=True, return_counts=True)[-1]

    #dividing the highest count by the sum (total)                         
    density_ben = counts_ben.max() / counts_ben.sum()
    density_mal = counts_mal.max() / counts_mal.sum()
    
    return density_ben, density_mal