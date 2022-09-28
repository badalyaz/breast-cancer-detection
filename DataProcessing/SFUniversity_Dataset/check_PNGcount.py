'''
    A simple mechanism for checking whether the PNG files generated are correct or not.
'''

from glob import glob
pngs = glob("/home/server/other_projects/breast_cancer/DATA_PATH/Data/SFUniversity/*/*/*/*/*/*/*/*/PNGFiles/*")
ljpegs = glob("/home/server/other_projects/breast_cancer/DATA_PATH/Data/SFUniversity/*/*/*/*/*/*/*/*/PNGFiles/../*.LJPEG")
pngs = list(set(pngs))


print(len(pngs),len(ljpegs))


missing = []
checked = []
duplicates = []
for ljpeg in ljpegs:
    ljpeg_filename = ljpeg[ljpeg.rfind("/")+1:ljpeg.rfind(".")]
    ljpeg_case_folder = ljpeg[:ljpeg.rfind("/PNGFiles")]

    b=False
    for png in pngs:
        png_filename = png[png.rfind("/")+1:png.rfind(".")]
        png_case_folder = png[:png.rfind("/PNGFiles")]
        if ljpeg_filename == png_filename and ljpeg_case_folder==png_case_folder:
            # print(f"{ljpeg_filename}:{png_filename}")
            if (png in checked):
                duplicates.append(png)
            else:
                checked.append(png)
            b=True
            break
    
    if not b:
        # missing.append(ljpeg_case_folder+"/PNGFiles/"+ljpeg_filename+".png")
        missing.append(ljpeg_case_folder+"/"+ljpeg_filename+".LJPEG")

print(f"These png files are missing --> {missing}\n\n")
print(f"These files have duplicates --> {duplicates}")