import numpy as np
import os
import re

def read_data(filename):
    with open(filename, 'r') as file:
        # Skip the first three lines
        for _ in range(3):
            next(file)
        
        # Read the rest of the file
        data = np.loadtxt(file)
    
    return data

def get_folder_paths(folder_path, folder_pattern):
    folder_list = [folder_path + f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f)) and re.search(folder_pattern, f)]
    folder_list.sort(key=os.path.getctime)
    return folder_list

def extract_number(filename):
    match = re.search(r'C\[(\d+)\]-avg-\.plt', filename)
    return int(match.group(1)) if match else float('inf') 

def sort_filenames(filenames):
    return sorted(filenames, key=extract_number)

def get_filenames(file_path, file_pattern):
    file_list = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f)) and re.search(file_pattern, f)]
    file_list = sort_filenames(file_list)
    return file_list