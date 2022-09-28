import os
import pandas
from tqdm import tqdm
from datetime import datetime

################################ Pre Defining paths and server info ####################################
################################        CHANGE ONLY THIS PART       ####################################
root = "C:\\Users\\data_science\\Desktop\\Projects\\BreastCancer\\SFUniversity"
paths = pandas.read_csv(f"{root}\\SFU_case_paths.csv")["paths"]

host = "hostname@255.255.255.255"
host_passw = "Password"
#########################################################################################################


os.chdir(f"{root}\\")
if os.path.exists(f"{root}\\Download"):
    os.system("rd /s/q  Download")

os.mkdir("Download")
ind = 1107
end_ind = 1360
for path in tqdm(paths[ind:end_ind]):
    
    ### Get case folder name
    filename = path[path.rfind("/")+1:]
    if os.path.exists(f"{root}\\Download"):
        os.system("rd /s/q  Download")
    os.mkdir("Download")

    
    try:
        ### Get case folder from the server
        os.system(f'pscp -pw {host_passw} -r {host}:{path} {root}\\Download  ') 
        with open("log.txt","a",encoding="utf-8") as log_file:
            log_file.write(f'{filename}({ind})\n\t→ Got from {host} ||:::|| {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n')
        
        ### Run Matlab code via batch file
        os.system(f"{root}\\runMatlab.bat")
        with open("log.txt","a",encoding="utf-8") as log_file:
            log_file.write(f"\t→ Converted  ||:::|| {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        
        ### Copy PNGFiles to server
        os.system(f'pscp -pw {host_passw} -r {root}\\Download\\{filename}\\PNGFiles server@192.168.5.8:{path}   ') 
        with open("log.txt","a",encoding="utf-8") as log_file:    
            log_file.write(f"\t→ Moved [{filename}\\PNGFiles] to {host} ||:::|| {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")

    except:
        with open("log.txt","a",encoding="utf-8") as log_file:
            log_file.write(f"{filename}({ind})\n\t→ SOMETHING WENT WRONG!!!! ||:::|| {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")

    ind+=1
