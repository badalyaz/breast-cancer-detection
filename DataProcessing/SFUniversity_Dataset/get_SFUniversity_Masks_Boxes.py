'''


This script goes through all the SFUniversity direction and finds all the annotation files.
The files have .ics and .OVERLAY extensions.
The annotations can have some issues in them as they are handwritten and don't have fixed structure
    SO ...
        The code is tried to be made maximally flexible for the data, but anyways ...
        Be very careful, the code is very sensitive to small mistakes in the data



'''
import matplotlib.pyplot as plt
import os
import pickle
import numpy
from glob import glob
from tqdm import tqdm

def decode_overlay_poly(poly):
    
    ### Finding the polygon of mask
    res_dict = {}
    mask = []
    
    for i in poly:
        if i.isdigit():
            mask.append(int(i) )
        else:
            continue
    mask_rows = [mask[1]]
    mask_columns = [mask[0]]
    for i in range(3,len(poly)):
        code_val=int(poly[i])
        if code_val==0:
            mask_columns.append(mask_columns[i-3])
            mask_rows.append(mask_rows[i-3]-1)
        elif code_val == 1:
            mask_columns.append(mask_columns[i-3]+1)
            mask_rows.append(mask_rows[i-3]-1)
        elif code_val == 2:
            mask_columns.append(mask_columns[i-3]+1)
            mask_rows.append(mask_rows[i-3])
        elif code_val== 3:
            mask_columns.append(mask_columns[i-3]+1)
            mask_rows.append(mask_rows[i-3]+1)
        elif code_val== 4:
            mask_columns.append(mask_columns[i-3])
            mask_rows.append(mask_rows[i-3]+1)
        elif code_val== 5:
            mask_columns.append(mask_columns[i-3]-1)
            mask_rows.append(mask_rows[i-3]+1)
        elif code_val== 6:
            mask_columns.append(mask_columns[i-3]-1)
            mask_rows.append(mask_rows[i-3])
        else:
            mask_columns.append(mask_columns[i-3]-1)
            mask_rows.append(mask_rows[i-3]-1)
    
    ## Adding the mask polygon coordinates and bbox
    res_dict["mask_x"] = numpy.array(mask_columns,dtype=int)
    res_dict["mask_y"] = numpy.array(mask_rows,dtype=int)
    res_dict["bbox"] = [numpy.min(mask_columns),numpy.min(mask_rows),numpy.max(mask_columns),numpy.max(mask_rows)]

    return res_dict


def get_ics_annots(ics_path, img_ori):
    ## Open the .ics file
    '''
    0)ics_version 1.0
    1)filename C-0419-1
    2)DATE_OF_STUDY 6 6 1995
    3)PATIENT_AGE 74
    4)FILM 
    5)FILM_TYPE REGULAR
    6)DENSITY 3
    7)DATE_DIGITIZED 16 11 1998 
    8)DIGITIZER LUMISYS LASER
    9)SEQUENCE
    10)LEFT_CC LINES 4544 PIXELS_PER_LINE 2936 BITS_PER_PIXEL 12 RESOLUTION 50 OVERLAY
    11)LEFT_MLO LINES 4576 PIXELS_PER_LINE 2904 BITS_PER_PIXEL 12 RESOLUTION 50 OVERLAY
    12)RIGHT_CC LINES 4552 PIXELS_PER_LINE 2912 BITS_PER_PIXEL 12 RESOLUTION 50 NON_OVERLAY
    13)RIGHT_MLO LINES 4560 PIXELS_PER_LINE 2888 BITS_PER_PIXEL 12 RESOLUTION 50 NON_OVERLAY

    from this file we need the lines [3,5,6,8,{one of the 10-13 lines}]
    and from lines 10,11,12,13 we will select the one that matches the orientation of the image
'''
    with open(ics_path,"r") as f:
        #skipping the empty row at the begining of .ics files   [1:]     
        ics_file = f.readlines()[1:]

    orientation_rows = {
        "LEFT_CC":len(ics_file)-4,
        "LEFT_MLO":len(ics_file)-3,
        "RIGHT_CC":len(ics_file)-2,
        "RIGHT_MLO":len(ics_file)-1,
    }
    
    key_indices = [3,5,6,8,orientation_rows[img_ori]]

    ## Taking the necessary lines from the .ics file and splitting them with " "
    kvs = []    
    kvs.extend(ics_file[i][:-1].split(" ") for i in key_indices)

    ## As the frist 3 lines are in pairs, converting them to key:value with dict comprehendion
    ## taking the 4th line's value as an array as "DIGITIZER" is described with 2 values -> [name, lasr_type/density]
    res_dict = {}
    
    filename = (ics_path[ics_path.rfind("/")+1:ics_path.rfind(".")]).replace("-","_") + "." + img_ori

    res_dict["image_path"] = f"{ics_path[:ics_path.rfind('/')]}/PNGFiles/{filename}.png"
    res_dict.update({i[0]:(float(i[1]) if i[1].isdigit() else i[1]) for i in kvs[:3]})
    res_dict[kvs[3][0]] = kvs[3][1:]

    
    for key in range(1,len(kvs[4])-1,2):
        res_dict[kvs[4][key]] = float(kvs[4][key+1])
    res_dict["Has_Overlay"] = True if kvs[4][-1]=="OVERLAY" else False
    
    return res_dict


def get_overlay_annots(overlay_path):
    ## Error checking
    if not os.path.exists(overlay_path):
        raise FileNotFoundError()
    
    ## opening the annot file
    with open(overlay_path,"r") as f:
        file = f.readlines()    

    ## Split all the lines with " " and as a result
    # the keys   will be 0th,2nd,4th,... elements and
    # the values will be 1st,3rd,5th,... elements
    res_dict = {}

    res_dict["TOTAL_ABNORMALITIES"] = int(file[0][:-1].split(" ")[1])
    res_dict["ABNORMALITIES"] = {}

    range_end = 0
    for abnormality_i in range(res_dict["TOTAL_ABNORMALITIES"]):
        abn_id = f"ABNORMALITY_{abnormality_i+1}"
        res_dict["ABNORMALITIES"][abn_id] = {}

        # Remove extra charachters "\n" and get an array of kvs 
        kv = []
        boundary_indices = [file[(abnormality_i+1)*8-1:(abnormality_i+1)*8+3].index("BOUNDARY\n")+(abnormality_i+1)*8]
        
        range_start = range_end+2
        range_end = boundary_indices[0]
        for i in range(range_start,range_end-1): 
            a = file[i][:-1].split(" ")
            try:
                a.remove("")          
            except: None
            kv.extend(a)

        for i in range(0,len(kv)-1,2):
            res_dict["ABNORMALITIES"][abn_id][kv[i]]= int(kv[i+1]) if kv[i+1].isdigit() else kv[i+1]




        ind = 0
        while boundary_indices[ind]+2<len(file) and \
            "CORE" in file[boundary_indices[ind]+2]:    
            boundary_indices.append(boundary_indices[ind]+2)
            range_end+=1
            ind+=1

        ## improve the number of total_outlines from .overlay files
        ## because there are a lot of them that indicate a wrong number of outlines
        res_dict["ABNORMALITIES"][abn_id]["TOTAL_OUTLINES"] = len(boundary_indices)

        ## getting the mask(s)
        poly = [decode_overlay_poly(file[ind].split(" ")[:-1]) for ind in boundary_indices]
        res_dict["ABNORMALITIES"][abn_id]["masks"]=poly
        
    return res_dict




def get_all_overlays(sfu_path):
    all_paths = glob((sfu_path + "/*/*/*/*/*/*/*/*/*.LJPEG"))
    result_dict = {}
    q=0
    for path in tqdm(all_paths[:]):
        ## Getting the orientation of image from the path

        st = path[path.rfind("/"):path.rfind(".")]
        img_ori = st[st.find(".")+1:]
        filename = st[1:st.find(".")]
        ## Getting the path for .ics file
        ics_path = glob(path[:path.rfind("/")] + "/*.ics")[0]
        filename += "." + img_ori

        result_dict[filename] = get_ics_annots(ics_path,img_ori)

        if (result_dict[filename]["Has_Overlay"]):
            ##Changing LJPEG with OVERLAY in path
            # print(path)
            path = path.replace("LJPEG","OVERLAY")
            result_dict[filename].update(get_overlay_annots(path))

    return result_dict


def main():
    import pickle

    ### Get all the annotations and save the result dict into a dict.pickle file
    with open('/home/server/other_projects/breast_cancer/DATA_PATH/Data/SFUniversity/Annotations.pickle', 'wb') as handle:
        pickle.dump(get_all_overlays("/home/server/other_projects/breast_cancer/DATA_PATH/Data/SFUniversity"),
                    handle, protocol=pickle.HIGHEST_PROTOCOL)

    ### An example of reading the dict.pickle file
    with open('/home/server/other_projects/breast_cancer/DATA_PATH/Data/SFUniversity/Annotations.pickle', 'rb') as handle:
        res = pickle.load(handle)

    # import rich
    # rich.print(res[list(res.keys())[0]])
    
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    plt_count = (5,5)
    plt.figure(figsize=[30 for i in range(2)])
    img_ind = 0
    for i in range(plt_count[0]):
        for j in range(plt_count[1]):            
            key = list(res.keys())[img_ind] 

            while (not os.path.exists(res[key]["image_path"])):
                img_ind+=1
                key = list(res.keys())[img_ind]       

            img = plt.imread(res[key]["image_path"])             
            ax = plt.subplot2grid(plt_count, (i,j))
            ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)   
            ax.set_title(key, fontsize=5)            
            ax.imshow(img)
            
            if "ABNORMALITIES" in res[key].keys():
                for abn in res[key]["ABNORMALITIES"]:
                    bbox = res[key]["ABNORMALITIES"][abn]["masks"][0]["bbox"]                
                    rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],
                                            linewidth=1, edgecolor='r', facecolor='none')
                    #add patches
                    ax.add_patch(rect)
                    mask = (res[key]["ABNORMALITIES"][abn]["masks"][0]["mask_x"],res[key]["ABNORMALITIES"][abn]["masks"][0]["mask_y"])
                    ax.plot(mask[0],mask[1],color="y")
            print(f"{img_ind+1}) on image {key}.png")
            img_ind+=1
    plt.savefig("/home/server/other_projects/breast_cancer/breast_cancer_gitlab/DataProcessing/SFUniversity_Dataset/test.png")


if __name__ == '__main__':
    main()
    
