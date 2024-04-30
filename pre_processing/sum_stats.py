from pathlib import Path
from PIL import Image
import os

DATA_PATH = '/Users/gabrielbarrett/Code/ML/xray/data/chest_xray'
METRICS = ["widths", "heights", "pixels_sq", "ratios"]

# Getting the Path where the x-ray images are stored

current_dir = Path.cwd()

# Get the parent directory of the current file
parent_dir = current_dir.parent

# Join the parent directory with the name of the adjacent folder
data_path = parent_dir.joinpath('data/chest_xray')


# test bottom path: /Users/gabrielbarrett/Code/ML/xray/data/chest_xray/test/NORMAL

def compile_stats(data_path):
    sum_stats = {}
    for folder_l1 in os.listdir(data_path):     
        #iterating through test, train, and val
        if folder_l1 == '.DS_Store':
            continue
        fl1 = os.path.join(data_path, folder_l1)
        
        # checking if it is a folder
        if not os.path.isdir(fl1):
            continue
        sum_stats[folder_l1] = {}
        for folder_l2 in os.listdir(fl1):
            #iterating through Normal and Pnemonia
            if folder_l2 == '.DS_Store':
                continue
            fl2 = os.path.join(fl1, folder_l2)
            if not os.path.isdir(fl2):
                continue
            sum_stats[folder_l1][folder_l2] = {}

            for filename in os.listdir(fl2):
                #iterating through images
                if filename == '.DS_Store':
                    continue
                f = os.path.join(fl2, filename)
                # checking if it is a file
                if not os.path.isfile(f):
                    continue
                
                
                image_obj = Image.open(f)

                #initializing empty lists of summary statistics
                for m in METRICS:
                    sum_stats[folder_l1][folder_l2][m] = sum_stats[folder_l1][folder_l2].get(m, [])

                #fillin in the summary statistics
                sum_stats[folder_l1][folder_l2]["widths"].append(image_obj.width)
                sum_stats[folder_l1][folder_l2]["heights"].append(image_obj.height)
                #calculating square pixels
                square_pixels = int(image_obj.height) * int(image_obj.width)
                sum_stats[folder_l1][folder_l2]["pixels_sq"].append(square_pixels)

                sum_stats[folder_l1][folder_l2]["ratios"].append(int(image_obj.width)/int(image_obj.height))
                del image_obj

    return sum_stats

