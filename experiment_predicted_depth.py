import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import hoho
from pathlib import Path
import argparse
from HouseData import HouseData
import ipdb
import time
from tqdm import tqdm

def get_iterable_hoho_dataset():
    data_dir = Path('./data/data')
    split = 'train'
    hoho.LOCAL_DATADIR = hoho.setup(data_dir)
    dataset = hoho.get_dataset(decode=None, split=split, dataset_type='webdataset')
    dataset = dataset.map(hoho.decode)
    iterable_dataset = iter(dataset)
    return iterable_dataset

# I want to see if we can use the monocular depth approroately scaled to filter out the 
# incorrectly triangulated points

if __name__ == "__main__":

    iterable_dataset = get_iterable_hoho_dataset()
    
    weds = []
    times = []
    for i in tqdm(range(100)):
        sample = next(iterable_dataset)
        
        start = time.time()
        house_data = HouseData(sample)

        house_data.get_sfm_pca()

        house_data.get_2d_corners()
        print("Obtained 2D corners")
        
        house_data.get_all_corners_using_monocular_depths()
        print("Obtained all corners using monocular depths")
        
        house_data.triangulate_all_2d_corner_pairs()
        print("Triangulated all 2D corner pairs")
        
        house_data.merge_triangulated_monocular_corners()
        
        house_data.get_edges(method='no_edges')

        house_data.compute_metric()
        end = time.time()
        print("Time taken for one house: ", end-start)
        times.append(end-start)
        weds.append(house_data.wed)
        # ipdb.set_trace()
        print("Running mean WED: ", np.mean(weds))
        print("Running median WED: ", np.median(weds))
    
    print(" -------------------------------------------- ")
    print("Average time taken for one house: ", np.mean(times))
    print("Average WED for 500 houses: ", np.mean(weds))
    print(" -------------------------------------------- ")
    print("Total time taken for 500 houses: ", np.sum(times))
    print("Median wed for 500 houses: ", np.median(weds))
