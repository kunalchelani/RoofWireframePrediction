import numpy as np
from utils import triangulate_multiview_algebraic_error
import open3d as o3d

def triangulate_pair_old(Ks, Rs, ts, vertices, debug_visualize = None,
                                                                   gt_lines_o3d = None,
                                                                   segmented_images = None,
                                                                   house_pts = None, 
                                                                   dist_thresh_house_pts = None):

    triangulated_points = []
    vertex_types = []
    pair_inds = []
    proj_mat0 = np.dot(Ks[0], np.hstack((Rs[0], ts[0].reshape(3,1))))
    proj_mat1 = np.dot(Ks[1], np.hstack((Rs[1], ts[1].reshape(3,1))))
    for i, v1 in enumerate(vertices[0]):
        for j, v2 in enumerate(vertices[1]):
            if v1['type'] == v2['type']:
                
                X,reprojection_errs  = triangulate_multiview_algebraic_error([proj_mat0, proj_mat1], [v1['xy'], v2['xy']])
                if house_pts is not None:
                    min_dist_house_pts = np.min(np.linalg.norm(house_pts - X.reshape(1,3), axis = 1))
                    if min_dist_house_pts > dist_thresh_house_pts[v1['type']]:
                        continue

                if np.sum(reprojection_errs) < 10:
                    # if verbose:
                    #     d1 = np.dot(Rs[0][2, :], X) + ts[0][2]
                    #     d2 = np.dot(Rs[1][2, :], X) + ts[1][2]
                    #     print(f"Depths in cameras {d1} and {d2}")
                    #     print(f"Reprojection errors: {reprojection_errs[0]} and {reprojection_errs[1]}")
                    pair_inds += [(i,j)]
                    triangulated_points += [X]                
                    vertex_types += [v1['type']]

    return triangulated_points, vertex_types, pair_inds

def triangulate_from_viewpoints(Ks, Rs, ts, vertices, debug_visualize = False, gt_lines_o3d = None, segmented_images = None, house_pts = None, 
                                                                   dist_thresh_house_pts = 100):
    if debug_visualize:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(gt_lines_o3d)
        vis.run()
        vis.destroy_window()

    print(len(Ks), len(Rs), len(ts), len(vertices))
    assert len(Ks) == len(Rs) == len(ts) == len(vertices)
    triangulated_points = []
    vertex_types = []
    pair_inds = []
    num_views = len(Ks)
    for i in range(num_views):
        for j in range(i+1, num_views):
            Ks_ = [Ks[i], Ks[j]]
            Rs_ = [Rs[i], Rs[j]]
            ts_ = [ts[i], ts[j]]
            if segmented_images is not None:
                segmented_images_ = [segmented_images[i], segmented_images[j]]
            triangulated_points_, vertex_types_, pair_inds_ = triangulate_pair_old(Ks_, Rs_, ts_, [vertices[i], vertices[j]],
                                                                   debug_visualize = debug_visualize,
                                                                   gt_lines_o3d = gt_lines_o3d,
                                                                   segmented_images = segmented_images_,
                                                                   house_pts = house_pts, 
                                                                   dist_thresh_house_pts = dist_thresh_house_pts,
                                                                   )
            triangulated_points += triangulated_points_
            vertex_types += vertex_types_
            pair_inds_ = [(i, j, pair_ind) for pair_ind in pair_inds_]
            pair_inds += pair_inds_


    return triangulated_points, vertex_types, pair_inds

def get_monocular_depths_at(monocular_depth, K, R, t, positions, scale = 0.32, max_z = 2000):

    # ipdb.set_trace()
    # Get the positions we need to look in the depth images
    uv = np.hstack((positions, np.ones((positions.shape[0], 1)))).astype(np.int32)

    # Sample the depths at these points
    depths =  monocular_depth[uv[:,1], uv[:,0]]

    # Get the 3D points
    Kinv = np.linalg.inv(K)
    
    xyz = np.dot(Kinv, uv.T).T * np.array(depths.reshape(-1,1)) * scale
    mask = (xyz[:,2] < max_z).T

    xyz = np.dot(R.T, xyz.T - t.reshape(3,1))

    return xyz.T, mask