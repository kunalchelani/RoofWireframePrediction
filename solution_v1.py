import hoho
from hoho import *

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# from collections import Counter
# from tqdm import tqdm
import webdataset as wds 
import open3d as o3d
# from handcrafted_baseline_submission.handcrafted_solution import *
from utils import process_points, get_triangulated_corners, get_edges_with_support
from hoho import vis
import ipdb
# from PIL import Image as PImage
import argparse

from hoho.read_write_colmap import read_cameras_binary, read_images_binary, read_points3D_binary
from hoho.color_mappings import gestalt_color_mapping, ade20k_color_mapping
from hoho import compute_WED
import cv2
from tqdm import tqdm
import time
import os

def solution_hc(data, process_points_intermediate = False, merge_threshold = 20, save_as_o3d = False):

    key = data['__key__']
    gestalt_segmented_images = data['gestalt']
    depth_images = data['depthcm']
    Ks = data['K']
    Rs = data['R']
    ts = data['t']
    
    triangulated_corners, triangulated_corner_classes = get_triangulated_corners(gestalt_segmented_images, depth_images, Ks, Rs, ts)

    if process_points_intermediate:
        triangulated_corners_merged, triangulated_corner_classes_merged = process_points(triangulated_corners, triangulated_corner_classes, merge = True, merge_threshold = merge_threshold)

    pred_edges_merged = get_edges_with_support(triangulated_corners_merged, triangulated_corner_classes_merged, gestalt_segmented_images, Ks, Rs, ts)
    pred_edges = get_edges_with_support(triangulated_corners, triangulated_corner_classes, gestalt_segmented_images, Ks, Rs, ts)
    # Save the o3d pointcloud and gt_lineset to ply files
    data_name = f"house_{key}"
    
    computed_wed_merged = compute_WED(triangulated_corners_merged, pred_edges_merged, data['wf_vertices'], data['wf_edges'])
    computed_wed = compute_WED(triangulated_corners, pred_edges, data['wf_vertices'], data['wf_edges'])

    print(f"Computed WED for {data_name} is {computed_wed}")
    print(f"Computed WED for {data_name} after merging is {computed_wed_merged}")
    
    if save_as_o3d:
        
        save_dir = "data/output/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        import o3d_utils

        predicted_wf_o3d = o3d_utils.get_wireframe_o3d(triangulated_corners_merged, pred_edges_merged)
        try:
            o3d.io.write_line_set(f"data/output/{data_name}_predicted_wf_merged.ply", predicted_wf_o3d, write_ascii=True)
        except:
            print("Error in writing predicted_wf")
        
        triangulated_pts_o3d = o3d_utils.get_triangulated_pts_o3d_pc(triangulated_corners_merged, triangulated_corner_classes_merged, gestalt_color_mapping)
        o3d.io.write_point_cloud(f"data/output/{data_name}_triangulated_pts_merged.ply", triangulated_pts_o3d, write_ascii=True)
        
        o3d_gt_lines = o3d_utils.get_wireframe_o3d(data['wf_vertices'], data['wf_edges'])
        o3d.io.write_line_set(f"data/output/{data_name}_gt_lines.ply", o3d_gt_lines, write_ascii=True)

    return key, triangulated_corners, pred_edges, computed_wed, computed_wed_merged

def main(args):
    from concurrent.futures import ProcessPoolExecutor

    num_houses = args.num_houses
    data_dir = Path('./data/')
    split = 'all'
    hoho.LOCAL_DATADIR = hoho.setup(data_dir)
    dataset = hoho.get_dataset(decode=None, split='all', dataset_type='webdataset')
    dataset = dataset.map(hoho.decode)
    iterable_dataset = iter(dataset)
    weds = []
    weds_merged = []
    solution = []
    if args.parallel:
        with ProcessPoolExecutor(max_workers=2) as pool:
            results = []
            for i in tqdm(range(args.num_houses)):
                # replace this with your solution
                sample = next(iterable_dataset)
                results.append(pool.submit(solution_hc, sample, 
                                           process_points_intermediate = args.process_points_intermediate,
                                           merge_threshold = args.merge_threshold,
                                           save_as_o3d=args.save_as_o3d))
                
            for i, result in enumerate(results):
                
                key, pred_vertices, pred_edges, computed_wed, computed_wed_merged = result.result()
                weds.append(computed_wed)
                weds_merged.append(computed_wed_merged)
                solution.append({
                                '__key__': key, 
                                'wf_vertices': pred_vertices,
                                'wf_edges': pred_edges
                        })
    
    else:
        for i in tqdm(range(args.num_houses)):
            # replace this with your solution
            sample = next(iterable_dataset)
            key, pred_vertices, pred_edges, computed_wed, computed_wed_merged = solution_hc(sample, 
                                                                                            process_points_intermediate = args.process_points_intermediate,
                                                                                            merge_threshold = args.merge_threshold,
                                                                                            save_as_o3d=args.save_as_o3d)
            weds.append(computed_wed)
            weds_merged.append(computed_wed_merged)
            solution.append({
                            '__key__': key, 
                            'wf_vertices': pred_vertices,
                            'wf_edges': pred_edges
                        })
    
    weds = np.array(weds)
    weds_merged = np.array(weds_merged)

    print(f"Mean WED for {num_houses} houses is {np.mean(weds)}")
    print(f"Mean WED merged for {num_houses} houses is {np.mean(weds_merged)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_as_o3d', action = 'store_true', help = 'Save the output as o3d files')
    parser.add_argument('--num_houses', type = int, default = 100, help = 'Number of houses to process')
    parser.add_argument('--parallel', action = 'store_true', help = 'Run the solution in parallel')
    parser.add_argument('--process_points_intermediate', action = 'store_true', help = 'Process the points intermediate')
    parser.add_argument('--merge_threshold', type = float, default = 20, help = 'Process the points intermediate')
    args = parser.parse_args()

    main(args)