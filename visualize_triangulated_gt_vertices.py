import open3d as o3d
import numpy as np
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize triangulated ground truth vertices')
    parser.add_argument('--show_gt', action='store_true', help='Show ground truth lines')
    parser.add_argument('--show_predicted', action='store_true', help='Show predicted lines')
    parser.add_argument('--show_triangulated', action='store_true', help='Show triangulated points')
    parser.add_argument('--house_number', type=str, help='Number of the house to plot')
    args = parser.parse_args()
    house_number = args.house_number

    base_path = "data/output"

    fnames = os.listdir(base_path)
    house_numbers = set([fname.split('_')[1] for fname in fnames])
    house_numbers = sorted(list(house_numbers))
    house_number = house_numbers[int(house_number)]
    print(f"Visualizing house {house_number}")
    
    triangulated_pts_ply_path = base_path + f"/house_{house_number}_triangulated_pts_merged.ply"
    gt_lines_ply_path = base_path + f"/house_{house_number}_gt_lines.ply"
    triangulated_pts = o3d.io.read_point_cloud(triangulated_pts_ply_path)
    gt_lines = o3d.io.read_line_set(gt_lines_ply_path)

    predicted_wf_ply_path = base_path + f"/house_{house_number}_predicted_wf_merged.ply"
    predicted_wf = o3d.io.read_line_set(predicted_wf_ply_path)
    predicted_wf.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]] * len(predicted_wf.lines)))

    geoms_show = []
    if args.show_gt:
        geoms_show.append(gt_lines)
    if args.show_predicted:
        geoms_show.append(predicted_wf)
    if args.show_triangulated:
        geoms_show.append(triangulated_pts)

    o3d.visualization.draw_geometries(geoms_show)
