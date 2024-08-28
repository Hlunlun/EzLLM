import os
import time
import shutil


ENZYME_TYPES = ['protease','amylase','lipase','mannanase','cellulase', 'pectinase', 'others']


def clean_dir(directory_path, days_threshold = 30, threshold = False):
    current_time = time.time()
    
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)

        def clean(item, path):        
            if os.path.isfile(path):
                os.remove(item_path)
                print(f"Removed file: {item}")
            elif os.path.isdir(path):
                shutil.rmtree(path)
                print(f"Removed directory: {item}")
        
        creation_time = os.path.getctime(item_path)
        if threshold == True:
            if (current_time - creation_time) // (24 * 3600) >= days_threshold:
                clean(item, item_path)
        else:
            clean(item, item_path)