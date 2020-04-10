import pickle
import os
from os import walk
#from save_load import *

cache_dir = "./cache/"

def load_cache(path):
    path = cache_dir + path
    print("Loading cache from file: ", path)
    try:
        with open(path, 'rb') as f:
            cache = pickle.load(f)
    except:
        cache = None
    return cache

def save_cache(obj, path):
    path = cache_dir + path
    print("Saving cache to file: ", path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def get_list_of_files(path):
    path = cache_dir + path
    list_of_files = []
    for (dirpath, dirnames, filenames) in walk(path):
        for file in filenames:
            list_of_files.append(file)
    for i in range(len(list_of_files)):
        print(i, ": ", list_of_files[i])
    wait = True
    while wait:
        elem = input("Which file do you want to load?\npress enter to quit\n")
        try:
            res = int(elem)
            if res >= 0 and res < len(list_of_files):
                wait = False
            else:
                print("Nb out of range: ", res)
        except:
            if elem == "":
                print("No file loaded")
                return None
    return list_of_files[res]
