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
import argparse
from o3d_utils import visualize_final_solution

def get_iterable_hoho_dataset():
    # data_dir = Path('/local/kunal/lines_localize/challenge/data/data/')
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type = str, default = "no_edges")
    parser.add_argument("--num_houses", type = int, default = 100)
    args = parser.parse_args()

    iterable_dataset = get_iterable_hoho_dataset()
    dist_thresh_house_pts_monocular = {'apex' : 250, 'eave_end_point' : 250,  'flashing_end_point' : 250}
    weds = {}
    times = []
    for i in tqdm(range(args.num_houses)):
        sample = next(iterable_dataset)
        
        start = time.time()
        house_data = HouseData(sample)

        house_data.get_2d_corners()
        print("Obtained 2D corners")

        house_data.process_sfm_pc()

        house_data.get_monocular_depths_from_sfm_intersection()

        house_data.get_all_corners_using_monocular_depths(dist_thresh_house_pts_monocular = dist_thresh_house_pts_monocular)
        
        house_data.triangulate_all_2d_corner_pairs_new()
        print("Triangulated all 2D corner pairs")

        triangulated_corners = np.array([corner['xyz'] for corner in house_data.triangulated_corners])
        monocular_est_corners = np.array(house_data.monocular_est_corners)

        # house_data.merge_triangulated_monocular_corners_new()
        house_data.merge_triangulated_monocular_corners_keep_all()
        
        house_data.get_num_sfm_points_within(200)

        house_data.get_lines_from_sfm_points(visualize = False)
            
        house_data.get_edges(method="new_hc")

        house_data.compute_metric()

        # house_data.plot_2d()
        # if house_data.wed > 2.2:
            
        #     house_data.plot_2d()

        #     # visualize the house
        #     triangulated_corners = np.array([corner['xyz'] for corner in house_data.triangulated_corners])
        #     monocular_est_corners = np.array(house_data.monocular_est_corners)

        #     visualize_final_solution(house_data.pred_wf_edges,
        #                              house_data.pred_wf_vertices,
        #                              house_data.pred_wf_vertices_classes,
        #                              house_data.gt_wf_edges,
        #                              house_data.gt_wf_vertices,
        #                             #  triangulated_corners,
        #                             #  monocular_est_corners,
        #                              None,
        #                              None,
        #                              house_data.house_pts)
            
        #     ipdb.set_trace()
            
        #     continue

        end = time.time()
        print("Time taken for one house: ", end-start)
        times.append(end-start)
        weds[house_data.house_key] = house_data.wed
        # ipdb.set_trace()
        print(f"Running mean WED: {np.mean(np.array(list(weds.values())))}")
        print("Running median WED: ", np.median(np.array(list(weds.values()))))

        # visualize the triangulated and monocular est points along with sfm_pts and gt_wireframe
        # visualize_final_solution(None,
        #                              house_data.pred_wf_vertices,
        #                              house_data.pred_wf_vertices_classes,
        #                              house_data.gt_wf_edges,
        #                              house_data.gt_wf_vertices,
        #                              all_init_triangulated = None,
        #                              all_init_monocular = None,
        #                              sfm_points = None)
        #                             #  monocular_est_corners,
                                    #  house_data.sfm_points)
        
        continue
        
        # house_data.get_all_corners_using_monocular_depths()
        print("Obtained all corners using monocular depths")

        house_data.triangulate_all_2d_corner_pairs()
        print("Triangulated all 2D corner pairs")
        
        house_data.merge_triangulated_monocular_corners()
        
        house_data.get_num_sfm_points_within(200)

        house_data.get_lines_from_sfm_points()
        
        house_data.get_edges(method="no_edges")

        house_data.compute_metric()
        # if house_data.wed > 2:
            
        #     house_data.plot_2d()

        #     # visualize the house
        #     triangulated_corners = np.array([corner['xyz'] for corner in house_data.triangulated_corners])
        #     monocular_est_corners = np.array(house_data.monocular_est_corners)
        #     visualize_final_solution(house_data.pred_wf_edges,
        #                              house_data.pred_wf_vertices,
        #                              house_data.pred_wf_vertices_classes,
        #                              house_data.gt_wf_edges,
        #                              house_data.gt_wf_vertices,
        #                              triangulated_corners,
        #                              monocular_est_corners,
        #                              house_data.sfm_points)
        #     continue

        end = time.time()
        print("Time taken for one house: ", end-start)
        times.append(end-start)
        weds[house_data.house_key] = house_data.wed
        # ipdb.set_trace()
        print(f"Running mean WED: {np.mean(np.array(list(weds.values())))}")
        print("Running median WED: ", np.median(np.array(list(weds.values()))))
    
    print(" -------------------------------------------- ")
    print("Average time taken for one house: ", np.mean(times))
    print(f"Average WED for {args.num_houses} houses: {np.mean(np.array(list(weds.values())))}")
    print(" -------------------------------------------- ")
    print(f"Total time taken for {args.num_houses} houses: ", np.sum(times))
    print(f"Median wed for {args.num_houses} houses: {np.median(np.array(list(weds.values())))}")

    # Save the dictionary of WEDs
    np.save(f"wed_{args.experiment_name}.npy", weds)
