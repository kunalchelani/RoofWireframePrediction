import numpy as np
# import torch
import cv2
from hoho.color_mappings import gestalt_color_mapping
import matplotlib.pyplot as plt
import ipdb
import open3d as o3d
import os
import time
import o3d_utils
import matplotlib
# matplotlib.use('GTK4Agg')

def get_vertices_from_gestalt(segim, color_threshold = 10):
    """
    Get the vertices of the gestalt from the segmented image
    :param segim: segmented image
    :param gestalt_color_mapping: color mapping of the gestalt
    :return: vertices of the gestalt
    """
    # Types of vertices - apex and eave_end_point

    vertices = []
    
    for vtype in ['apex', 'eave_end_point', 'flashing_end_point']:
        gest_seg_np = np.array(segim)
        apex_color = np.array(gestalt_color_mapping[vtype])
        apex_mask = cv2.inRange(gest_seg_np,  apex_color-color_threshold/2, apex_color+color_threshold/2)
        # apply the max filter to merge close by clusters
        apex_mask = cv2.dilate(apex_mask, np.ones((5,5), np.uint8), iterations=1)
        if apex_mask.sum() > 0:
            output = cv2.connectedComponentsWithStats(apex_mask, 8, cv2.CV_32S)
            (numLabels, labels, stats, centroids) = output
            stats, centroids = stats[1:], centroids[1:]
        
            # Add the centroid of the gestalt to the list of vertices
            for i, centroid in enumerate(centroids):
                vertices.append({"xy": centroid, "type": vtype})
            
    return vertices

def plot_centroids(vertices, gest_seg_np):
    """
    Plot the centroids of the gestalt on the segmented image
    :param vertices: vertices of the gestalt
    :param gest_seg_np: segmented image
    :return: image with the centroids plotted
    """
    plt.figure()
    plt.imshow(gest_seg_np)
    for vertex in vertices:
        marker = '^' if vertex["type"] == 'apex' else 'x'
        plt.scatter(vertex["xy"][0], vertex["xy"][1], marker=marker, c='black', s=50)

def triangulate_multiview_algebraic_error(cams, points_2d):
    # input : cams - list of camera matrices
    # each cam in cams is a 3x4 P = K[R | t]
    # pts - list of homogenous 2D points
    assert len(cams) == len(points_2d)
    A = np.zeros((2*len(cams), 4))
    for i in range(len(cams)):
        A[2*i, :] = points_2d[i][0]*cams[i][2, :] - cams[i][0, :]
        A[2*i+1, :] = points_2d[i][1]*cams[i][2, :] - cams[i][1, :]

    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1, :]
    X = X/X[-1]
    
    projections = [np.dot(cam, X) for cam in cams]
    reprojection_diffs = np.array([(projections[i]/projections[i][-1])[:2] - points_2d[i] for i in range(len(cams))])
    reprojection_errs = np.linalg.norm(reprojection_diffs, axis=1)
    return X[:3], reprojection_errs

def triangulate_pair(Ks, Rs, ts, vertices, 
                     verbose = False,
                     debug_visualize = False,
                     gt_lines_o3d = None,
                     segmented_images = None,
                     reprojection_err_thresh = 10):

    triangulated_points = []
    vertex_types = []
    proj_mat0 = np.dot(Ks[0], np.hstack((Rs[0], ts[0].reshape(3,1))))
    proj_mat1 = np.dot(Ks[1], np.hstack((Rs[1], ts[1].reshape(3,1))))
    
    # order the vertices, proj_mat etc by number of vertices
    if len(vertices[0]) > len(vertices[1]):
        vertices_less = vertices[1]
        vertices_more = vertices[0]
        proj_mat_less = proj_mat1
        proj_mat_more = proj_mat0
        Rs_less = Rs[1]
        Rs_more = Rs[0]
        ts_less = ts[1]
        ts_more = ts[0]
    else:
        vertices_less = vertices[0]
        vertices_more = vertices[1]
        proj_mat_less = proj_mat0
        proj_mat_more = proj_mat1
        Rs_less = Rs[0]
        Rs_more = Rs[1]
        ts_less = ts[0]
        ts_more = ts[1]

    for i,v1 in enumerate(vertices[0]):
        for j, v2 in enumerate(vertices[1]):
            if v1['type'] == v2['type']:
                
                X,reprojection_errs  = triangulate_multiview_algebraic_error([proj_mat0, proj_mat1], [v1['xy'], v2['xy']])
                errors_sum = np.sum(reprojection_errs)
                if errors_sum < reprojection_err_thresh:
                    if verbose:
                        d1 = np.dot(Rs[0][2, :], X) + ts[0][2]
                        d2 = np.dot(Rs[1][2, :], X) + ts[1][2]
                        print(f"Depths in cameras {d1} and {d2}")
                        # format reprojection errors to 2 decimal places
                        reprojection_errs = [round(err, 2) for err in reprojection_errs]
                        print(f"Reprojection errors: {reprojection_errs[0]} and {reprojection_errs[1]}")
                    
                    triangulated_points.append(X)
                    vertex_types.append(v1['type'])

    # errors_min = {i:0 for i in range(len(vertices_less))}
    # triangulations_min = {i:None for i in range(len(vertices_less))}
    # vertex_classes = {i:None for i in range(len(vertices_less))}
    
    # potential_triangulations = []
    # potential_vertex_classes = []
    
    # for i,v1 in enumerate(vertices_less):
    #     for j, v2 in enumerate(vertices_more):
    #         if v1['type'] == v2['type']:
                
    #             X,reprojection_errs  = triangulate_multiview_algebraic_error([proj_mat_less, proj_mat_more], [v1['xy'], v2['xy']])
    #             errors_sum = np.sum(reprojection_errs)
    #             if errors_sum < reprojection_err_thresh:
    #                 if verbose:
    #                     d1 = np.dot(Rs_less[2, :], X) + ts_less[2]
    #                     d2 = np.dot(Rs_more[2, :], X) + ts_more[2]
    #                     print(f"Depths in cameras {d1} and {d2}")
    #                     # format reprojection errors to 2 decimal places
    #                     reprojection_errs = [round(err, 2) for err in reprojection_errs]
    #                     print(f"Reprojection errors: {reprojection_errs[0]} and {reprojection_errs[1]}")
                    
    #                 potential_triangulations.append(X)
    #                 potential_vertex_classes.append(v1['type'])

    #                 if errors_min[i] < errors_sum:
    #                     errors_min[i] = errors_sum
    #                     triangulations_min[i] = X
    #                     vertex_classes[i] = v1['type']

                    # if debug_visualize:
                        # Need to first plot the image used to triangulate the point along with the marked 2D points

                        # plot the two images with the points
                        # fig, ax = plt.subplots(1,2)
                        # color = 'red' if errors_sum > 10 else 'green'
                        # ax[0].imshow(segmented_images[0])
                        # ax[0].scatter(v1['xy'][0], v1['xy'][1], c=color, s=50, marker='x')
                        # ax[1].imshow(segmented_images[1])
                        # ax[1].scatter(v2['xy'][0], v2['xy'][1], c=color, s=50, marker='x')
                        # ax[0].title.set_text(f"Reprojection error: {reprojection_errs[0]}")
                        # ax[1].title.set_text(f"Reprojection error: {reprojection_errs[1]}")
                        # plt.show()

                        # o3d visualization of the points
                        # vis = o3d.visualization.Visualizer()
                        # add the geometry consisting of all triangulated points with the current point being in black
                        # triangulated_pts_o3d = o3d_utils.get_open3d_point_cloud(potential_triangulations, potential_vertex_classes)
                        # current_pt_o3d = o3d.geometry.PointCloud()
                        # current_pt_o3d.points = o3d.utility.Vector3dVector([X])
                        # current_pt_o3d.colors = o3d.utility.Vector3dVector([[0,0,0]])
                        # o3d.visualization.draw_geometries([triangulated_pts_o3d, current_pt_o3d, gt_lines_o3d])

    # triangulated_points = [triangulations_min[i] for i in range(len(vertices_less)) if triangulations_min[i] is not None]
    # vertex_types = [vertex_classes[i] for i in range(len(vertices_less)) if triangulations_min[i] is not None]

    # ipdb.set_trace()
    # plot the final set of triangulated points

    if len(triangulated_points) > 0:
        if debug_visualize:
            triangulated_pts_o3d = o3d_utils.get_open3d_point_cloud(triangulated_points, vertex_types)
            o3d.visualization.draw_geometries([triangulated_pts_o3d, gt_lines_o3d])
    else:
        print("No triangulated points found")

    return triangulated_points, vertex_types

def triangulate_pair_old(Ks, Rs, ts, vertices, debug_visualize = None,
                                                                   gt_lines_o3d = None,
                                                                   segmented_images = None):

    triangulated_points = []
    vertex_types = []
    proj_mat0 = np.dot(Ks[0], np.hstack((Rs[0], ts[0].reshape(3,1))))
    proj_mat1 = np.dot(Ks[1], np.hstack((Rs[1], ts[1].reshape(3,1))))
    for v1 in vertices[0]:
        for v2 in vertices[1]:
            if v1['type'] == v2['type']:
                
                X,reprojection_errs  = triangulate_multiview_algebraic_error([proj_mat0, proj_mat1], [v1['xy'], v2['xy']])
                if np.sum(reprojection_errs) < 10:
                    # if verbose:
                    #     d1 = np.dot(Rs[0][2, :], X) + ts[0][2]
                    #     d2 = np.dot(Rs[1][2, :], X) + ts[1][2]
                    #     print(f"Depths in cameras {d1} and {d2}")
                    #     print(f"Reprojection errors: {reprojection_errs[0]} and {reprojection_errs[1]}")
                    
                    triangulated_points += [X]                
                    vertex_types += [v1['type']]

    return triangulated_points, vertex_types

def triangulate_from_viewpoints(Ks, Rs, ts, vertices, debug_visualize = False, gt_lines_o3d = None, segmented_images = None):
    if debug_visualize:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(gt_lines_o3d)
        vis.run()
        vis.destroy_window()

    assert len(Ks) == len(Rs) == len(ts) == len(vertices)
    triangulated_points = []
    vertex_types = []
    num_views = len(Ks)
    for i in range(num_views):
        for j in range(i+1, num_views):
            Ks_ = [Ks[i], Ks[j]]
            Rs_ = [Rs[i], Rs[j]]
            ts_ = [ts[i], ts[j]]
            if segmented_images is not None:
                segmented_images_ = [segmented_images[i], segmented_images[j]]
            triangulated_points_, vertex_types_ = triangulate_pair_old(Ks_, Rs_, ts_, [vertices[i], vertices[j]],
                                                                   debug_visualize = debug_visualize,
                                                                   gt_lines_o3d = gt_lines_o3d,
                                                                   segmented_images = segmented_images_)
            triangulated_points += triangulated_points_
            vertex_types += vertex_types_


    return triangulated_points, vertex_types

def get_3d_points_from_vertex(vertex, K, R, t, depthim, scale = 1.0):
    # get 3D points from a vertex
    Kinv = np.linalg.inv(K)
    xys = vertex['xy']
    uv = np.array([xys[0], xys[1], 1])
    depth = depthim[int(xys[1]), int(xys[0])]
    xyz = np.dot(Kinv, uv) * depth * scale
    xyz = np.dot(R.T, xyz - t)
    return xyz

def depth_to_3d_points(K, R, t, depthim, scale = 1.0):
    # get 3D points from a depth image
    Kinv = np.linalg.inv(K)
    h, w = depthim.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    uv = np.vstack((x.ravel(), y.ravel(), np.ones_like(x.ravel())))
    xyz = np.dot(Kinv, uv) * np.array(depthim).ravel() * scale
    xyz = np.dot(R.T, xyz - t.reshape(3,1))
    return xyz.T

def merge_neighboring_points(points, triangulated_corner_classes, thresh):
    
    # Points of the same class lying within the thresh distance are merged into a sinlge point - mean of the neighboring points
    cluster_pt_map = {}
    pt_cluster_map = {}
    for i in range(len(points)):
        if i not in pt_cluster_map:
            cluster_num = len(cluster_pt_map)
            pt_cluster_map[i] = cluster_num
            cluster_pt_map[cluster_num] = [i]

        for j in range(i+1, len(points)):
            if j not in pt_cluster_map:
                if triangulated_corner_classes[i] == triangulated_corner_classes[j]:
                    dist = np.linalg.norm(points[i] - points[j])
                    if dist < thresh:
                        pt_cluster_map[j] = pt_cluster_map[i]
                        cluster_pt_map[pt_cluster_map[i]].append(j)
                    # else:
                        # print(dist)
    
    merged_pts = []
    merged_classes = []
    # ipdb.set_trace()

    for cluster, pts in cluster_pt_map.items():
        merged_pts.append(np.mean(points[pts], axis=0))
        merged_classes.append(triangulated_corner_classes[pts[0]])

    return np.array(merged_pts), merged_classes

def process_points(points, triangulated_corner_classes, merge = True, merge_threshold = 20, remove = False, append = False):

    if merge:
        # merge the points
        points = np.vstack(points)
        # merge points within 20 cm
        merged_points, merged_classes = merge_neighboring_points(points, triangulated_corner_classes, merge_threshold)

    return merged_points, merged_classes


def get_triangulated_corners(gestalt_segmented_images, Ks, Rs, ts, 
                             debug_visualize = False,
                             gt_lines_o3d = None):

    if debug_visualize:
        assert gt_lines_o3d is not None     
    
    vertices_all = []
    for segim in gestalt_segmented_images:
        gest_seg_np = np.array(segim)
        vertices = get_vertices_from_gestalt(gest_seg_np)
        # plot_centroids(vertices, gest_seg_np)
        vertices_all.append(vertices)

    # Need to add robutness here, some points are too far off.
    triangulated_vertices, vertex_types = triangulate_from_viewpoints(Ks, Rs, ts, vertices_all, 
                                                                      debug_visualize = debug_visualize,
                                                                      gt_lines_o3d = gt_lines_o3d,
                                                                      segmented_images = gestalt_segmented_images)
    
    return triangulated_vertices, vertex_types

# def appropriate_line_close(pt, gest_seg_np, class_i, class_j, patch_size = 10, thresh = 10):
#     # If class i and class j are both apex, then we are looking for a ridge
#     # If class i is apex and class j is eave, then we are looking for a rake or valley
#     # If class i is eave and class j is apex, then we are looking for a rake or valley
#     # If class i and class j are both eave, then we are looking for a eave
#     # If class i is flashing_end_point and class j is apex, then we are looking for rake
#     # If class i is flashing_end_point and class j is eave, then we are looking for rake
#     # Not dealing with other cases for now
#     ridge_color = np.array(gestalt_color_mapping['ridge'])
#     rake_color = np.array(gestalt_color_mapping['rake'])
#     eave_color = np.array(gestalt_color_mapping['eave'])
#     valley_color = np.array(gestalt_color_mapping['valley'])
#     mask = None
#     # get a window around the point
#     window = gest_seg_np[int(pt[1])-2:int(pt[1])+patch_size//2, int(pt[0])-2:int(pt[0])+patch_size//2]
#     if class_i == 'apex' and class_j == 'apex':
#         mask = cv2.inRange(window, ridge_color-thresh/2, ridge_color+thresh/2)
#     elif class_i == 'apex' and class_j == 'eave_end_point':
#         mask = cv2.inRange(window, rake_color-thresh/2, rake_color+thresh/2)
#     elif class_i == 'eave_end_point' and class_j == 'apex':
#         mask = cv2.inRange(window, rake_color-thresh/2, rake_color+thresh/2)
#     elif class_i == 'eave_end_point' and class_j == 'eave_end_point':
#         mask = cv2.inRange(window, eave_color-thresh/2, eave_color+thresh/2)
#     elif class_i == 'flashing_end_point' and class_j == 'apex':
#         mask = cv2.inRange(window, rake_color-thresh/2, rake_color+thresh/2)
#     elif class_i == 'flashing_end_point' and class_j == 'eave_end_point':
#         mask = cv2.inRange(window, rake_color-thresh/2, rake_color+thresh/2)
    
#     if (mask is not None) and (np.sum(mask) > 1):
#         return True
    
#     return False

# def get_edges_with_support(points_3d_coords, points_3d_classes, gestalt_segmented_images, Ks, Rs, ts, debug_visualize = False, house_number = "house"):
#     # For each edge, find the supporting points
#     edges = []
#     for i in range(len(points_3d_coords)):
#         for j in range(i+1, len(points_3d_coords)):
            
#             support_ims = 0
#             observed_ims = 0

#             for k in range(len(gestalt_segmented_images)):

#                 gest_seg_np = np.array(gestalt_segmented_images[k])
#                 K = Ks[k]
#                 R = Rs[k]
#                 t = ts[k]
                
#                 # Project the 3D points to the image plane
#                 proj_i = np.dot(K, (np.dot(R, points_3d_coords[i]) + t))
#                 proj_i = (proj_i/proj_i[2]).astype(np.int32)
#                 proj_i = proj_i[:2]
#                 proj_j = np.dot(K, (np.dot(R, points_3d_coords[j]) + t))
#                 proj_j = (proj_j/proj_j[2]).astype(np.int32)
#                 proj_j = proj_j[:2]
#                 # Check that both the projections are in the region of the class they belong to
#                 class_i = points_3d_classes[i]
#                 class_j = points_3d_classes[j]

#                 # First check if the projections are in the image
#                 if proj_i[0] < 0 or proj_i[0] >= gest_seg_np.shape[1] or proj_i[1] < 0 or proj_i[1] >= gest_seg_np.shape[0]:
#                     continue
#                 if proj_j[0] < 0 or proj_j[0] >= gest_seg_np.shape[1] or proj_j[1] < 0 or proj_j[1] >= gest_seg_np.shape[0]:
#                     continue

#                 # Get the 3x3 region around the projections, apply the mask of the class color +- threshold and check the sum of the mask
#                 # If the sum is greater than 5, then the point is in the region of the class
#                 color_i = np.array(gestalt_color_mapping[class_i])
#                 window_i = gest_seg_np[int(proj_i[1])-1:int(proj_i[1])+2, int(proj_i[0])-1:int(proj_i[0])+2]
#                 mask_i = cv2.inRange(window_i, color_i-5, color_i+5)

#                 color_j = np.array(gestalt_color_mapping[class_j])
#                 window_j = gest_seg_np[int(proj_j[1])-1:int(proj_j[1])+2, int(proj_j[0])-1:int(proj_j[0])+2]
#                 mask_j = cv2.inRange(window_j, color_j-5, color_j+5)

#                 if np.sum(mask_i) > 5 and np.sum(mask_j) < 5:
#                     continue

#                 line = np.linspace(proj_i, proj_j, 12)
#                 if np.linalg.norm(line[0] - line[-1]) < 50:
#                     continue

#                 observed_ims += 1

#                 # ipdb.set_trace()
#                 # plot the line and the projected points on top of the segmented image and show

#                 # plt.imshow(gest_seg_np)
#                 # plt.scatter(proj_i[0], proj_i[1], c='red', s=50)
#                 # plt.scatter(proj_j[0], proj_j[1], c='red', s=50)
#                 # plt.plot(line[:,0], line[:,1], c='black', lw=2)
#                 # plt.show()

#                 support_pts = 0
#                 for pt in line[1:-1]:
#                     if appropriate_line_close(pt, gest_seg_np, class_i,class_j, patch_size = 5):
#                         support_pts += 1
#                 # print("Support pts: ", support_pts)
#                 if support_pts >= 5:
#                     support_ims += 1

                

#             # print("Support ims: ", support_ims)
#             # print("Observed ims: ", observed_ims)

#             if observed_ims == 0:
#                 continue
            
#             if support_ims/observed_ims > 0.5:
#                 edges.append([i,j])    
    
#     return edges, None


def appropriate_line_close(pt, gest_seg_np, class_i, class_j, patch_size = 10, thresh = 10):
    # If class i and class j are both apex, then we are looking for a ridge
    # If class i is apex and class j is eave_end_point, then we are looking for a rake or step_flashing
    # If class i is eave_endpoint and class j is apex, then we are looking for a rake or step_flashing
    # If class i and class j are both eave_end_point, then we are looking for a eave
    # If class i is flashing_end_point and class j is flashing_end_point, then we are looking for flashing
    # If class i is flashing_end_point and class j is apex, then we are looking for rake
    # If class i is flashing_end_point and class j is eave, then we are looking for rake
    # Not dealing with other cases for now
    ridge_color = np.array(gestalt_color_mapping['ridge'])
    rake_color = np.array(gestalt_color_mapping['rake'])
    eave_color = np.array(gestalt_color_mapping['eave'])
    valley_color = np.array(gestalt_color_mapping['valley'])
    flashing_color = np.array(gestalt_color_mapping['flashing'])
    step_flashing_color = np.array(gestalt_color_mapping['step_flashing'])
    mask = None
    # get a window around the point
    window = gest_seg_np[int(pt[1])-patch_size//2:int(pt[1])+patch_size//2, int(pt[0])-patch_size//2:int(pt[0])+patch_size//2]
    if class_i == 'apex' and class_j == 'apex':
        mask = cv2.inRange(window, ridge_color-thresh/2, ridge_color+thresh/2)
    
    elif class_i == 'eave_end_point' and class_j == 'eave_end_point':
        mask = cv2.inRange(window, eave_color-thresh/2, eave_color+thresh/2)

    elif class_i == 'flashing_end_point' and class_j == 'flashing_end_point':
        mask = cv2.inRange(window, flashing_color-thresh/2, flashing_color+thresh/2)

    elif class_i == 'apex' and class_j == 'eave_end_point':
        mask1 = cv2.inRange(window, rake_color-thresh/2, rake_color+thresh/2)
        mask2 = cv2.inRange(window, step_flashing_color-thresh/2, step_flashing_color+thresh/2)
        mask = mask1 if np.sum(mask1) > np.sum(mask2) else mask2
    elif class_i == 'eave_end_point' and class_j == 'apex':
        mask1 = cv2.inRange(window, rake_color-thresh/2, rake_color+thresh/2)
        mask2 = cv2.inRange(window, step_flashing_color-thresh/2, step_flashing_color+thresh/2)
        mask = mask1 if np.sum(mask1) > np.sum(mask2) else mask2
    
    elif class_i == 'flashing_end_point' and class_j == 'apex':
        mask = cv2.inRange(window, rake_color-thresh/2, rake_color+thresh/2)
    elif class_i == 'apex' and class_j == 'flashing_end_point':
        mask = cv2.inRange(window, rake_color-thresh/2, rake_color+thresh/2)
    
    elif class_i == 'flashing_end_point' and class_j == 'eave_end_point':
        mask = cv2.inRange(window, rake_color-thresh/2, rake_color+thresh/2)
    elif class_i == 'eave_end_point' and class_j == 'flashing_end_point':
        mask = cv2.inRange(window, rake_color-thresh/2, rake_color+thresh/2)
    
    
    if (mask is not None) and (np.sum(mask) > 1):
        return True
    
    return False

def visualize_edge(pt1, pt2, gt_points, gt_edges, horizontal_components, vertical_component):

    o3d_line_direction = o3d.geometry.LineSet()
    o3d_line_direction.points = o3d.utility.Vector3dVector([pt1, pt2])
    o3d_line_direction.lines = o3d.utility.Vector2iVector([[0,1]])
    o3d_line_direction.colors = o3d.utility.Vector3dVector([[0.5,0,0.5],[0.5,0,0.5]])

    o3d_gt_wireframe = o3d.geometry.LineSet()
    o3d_gt_wireframe.points = o3d.utility.Vector3dVector(gt_points)
    o3d_gt_wireframe.lines = o3d.utility.Vector2iVector(gt_edges)

    o3d_horizontal_components = o3d.geometry.LineSet()
    o3d_horizontal_components.points = o3d.utility.Vector3dVector([pt1, pt1 + 200 * horizontal_components[0], pt1 + 200 * horizontal_components[1], pt1 + 200 * vertical_component[0]])
    o3d_horizontal_components.lines = o3d.utility.Vector2iVector([[0,1],[0,2],[0,3]])
    o3d_horizontal_components.colors = o3d.utility.Vector3dVector([[1,0,0],[0,1,0],[0,0,1]])

    o3d.visualization.draw_geometries([o3d_line_direction, o3d_gt_wireframe, o3d_horizontal_components])



def get_edges_with_support(points_3d_coords, points_3d_classes, gestalt_segmented_images, Ks, Rs, ts, horizontal_components = None, vertical_component = None,
                           gt_wireframe = None, debug_visualize = True, house_number = "house"):
    # For each edge, find the supporting points
    edges = []
    vertex_edge_count = {i:0 for i in range(len(points_3d_coords))}
    for i in range(len(points_3d_coords)):
        for j in range(i+1, len(points_3d_coords)):
            
            support_ims = 0
            observed_ims = 0

            class_i = points_3d_classes[i]
            class_j = points_3d_classes[j]
            
            line_3d = points_3d_coords[j] - points_3d_coords[i]
            line_3d_direction = line_3d/np.linalg.norm(line_3d)
            
            if horizontal_components is not None and vertical_component is not None:
                if class_i == "eave_end_point" and class_j == "eave_end_point":
                    # Should be along either of the major components pther than the vertical (with y componeent big)
                    # print(np.abs(horizontal_components.dot(line_3d_direction)))

                    # ipdb.set_trace()

                    # visulaize the line  and the gt wireframe
                    # o3d_line_direction = o3d.geometry.LineSet()
                    # o3d_line_direction.points = o3d.utility.Vector3dVector([points_3d_coords[i], points_3d_coords[j]])
                    # o3d_line_direction.lines = o3d.utility.Vector2iVector([[0,1]])
                    # o3d_line_direction.colors = o3d.utility.Vector3dVector([[1,0,0],[1,0,0]])

                    # o3d_gt_wireframe = o3d.geometry.LineSet()
                    # o3d_gt_wireframe.points = o3d.utility.Vector3dVector(gt_wireframe[0])
                    # o3d_gt_wireframe.lines = o3d.utility.Vector2iVector(gt_wireframe[1])

                    # o3d_horizontal_components = o3d.geometry.LineSet()
                    # o3d_horizontal_components.points = o3d.utility.Vector3dVector([points_3d_coords[i], points_3d_coords[i] + 200 * horizontal_components[0], points_3d_coords[i] + 200 * horizontal_components[1]])
                    # o3d_horizontal_components.lines = o3d.utility.Vector2iVector([[0,1],[0,2]])
                    # o3d_horizontal_components.colors = o3d.utility.Vector3dVector([[0,1,0],[0,0,1]])

                    # o3d.visualization.draw_geometries([o3d_line_direction, o3d_gt_wireframe, o3d_horizontal_components])

                    # ipdb.set_trace()
                    if np.abs(vertical_component.dot(line_3d_direction)) > 0.15:
                        continue
                    if np.min(np.abs(horizontal_components.dot(line_3d_direction))) > 0.15:
                        continue
                elif class_i == "apex" and class_j == "apex":
                    vertical_alignment = np.abs(vertical_component.dot(line_3d_direction))
                    if vertical_alignment > 0.1:
                        print("Rejecting edge due to vertical alignment")
                        print(vertical_alignment)
                        # visualize_edge(points_3d_coords[i], points_3d_coords[j], gt_wireframe[0], gt_wireframe[1], horizontal_components, vertical_component)
                        continue

                    min_horizontal_alignment = np.min(np.abs(horizontal_components.dot(line_3d_direction)))
                    if min_horizontal_alignment > 0.1:
                        print("Rejecting edge due to horizontal alignments")
                        print(np.abs(horizontal_components.dot(line_3d_direction)))
                        # print(horizontal_components)
                        # visualize_edge(points_3d_coords[i], points_3d_coords[j], gt_wireframe[0], gt_wireframe[1], horizontal_components, vertical_component)
                        continue

                    else:
                        edges.append([i,j])
                        vertex_edge_count[i] += 1
                        vertex_edge_count[j] += 1
                        continue


                elif ("eave_end_point" in [class_i,class_j]) and ("apex" in [class_i,class_j]):
                    if np.min(np.abs(horizontal_components.dot(line_3d_direction))) > 0.15:
                        continue

            for k in range(len(gestalt_segmented_images)):

                gest_seg_np = np.array(gestalt_segmented_images[k])
                K = Ks[k]
                R = Rs[k]
                t = ts[k]
                
                # Project the 3D points to the image plane
                proj_i = np.dot(K, (np.dot(R, points_3d_coords[i]) + t))
                proj_i = (proj_i/proj_i[2]).astype(np.int32)
                proj_i = proj_i[:2]
                proj_j = np.dot(K, (np.dot(R, points_3d_coords[j]) + t))
                proj_j = (proj_j/proj_j[2]).astype(np.int32)
                proj_j = proj_j[:2]


                # Check that both the projections are in the region of the class they belong to

                # First check if the projections are in the image
                if proj_i[0] < 0 or proj_i[0] >= gest_seg_np.shape[1] or proj_i[1] < 0 or proj_i[1] >= gest_seg_np.shape[0]:
                    continue
                if proj_j[0] < 0 or proj_j[0] >= gest_seg_np.shape[1] or proj_j[1] < 0 or proj_j[1] >= gest_seg_np.shape[0]:
                    continue

                # Get the 5x5 region around the projections, apply the mask of the class color +- threshold and check the sum of the mask
                # If the sum is greater than 5, then the point is in the region of the class
                check_projection_filter_size = 3
                try:
                    color_i = np.array(gestalt_color_mapping[class_i])
                    
                    min_y_i = max(int(proj_i[1]) - check_projection_filter_size//2,0)
                    max_y_i = min(int(proj_i[1])+ check_projection_filter_size//2, gest_seg_np.shape[0])

                    min_x_i = max(int(proj_i[0]) - check_projection_filter_size//2, 0)
                    max_x_i = min(int(proj_i[0]) + check_projection_filter_size//2, gest_seg_np.shape[1])

                    window_i = gest_seg_np[min_y_i:max_y_i, min_x_i:max_x_i]
                    mask_i = cv2.inRange(window_i, color_i-5, color_i+5)

                    color_j = np.array(gestalt_color_mapping[class_j])
                    min_y_j = max(int(proj_j[1]) - check_projection_filter_size//2,0)
                    max_y_j = min(int(proj_j[1]) + check_projection_filter_size//2, gest_seg_np.shape[0])

                    min_x_j = max(int(proj_j[0]) - check_projection_filter_size//2, 0)
                    max_x_j = min(int(proj_j[0]) + check_projection_filter_size//2, gest_seg_np.shape[1])

                    window_j = gest_seg_np[min_y_j:max_y_j, min_x_j:max_x_j]
                    mask_j = cv2.inRange(window_j, color_j-5, color_j+5)
                except:
                    ipdb.set_trace()

                # Why this condition? Seems like a bug. Should be if np.sum(mask_i) < 5 and np.sum(mask_j) < 5?
                if (np.sum(mask_i)) < 1 or (np.sum(mask_j) < 1):
                    # print(f"First point mask sum : {np.sum(mask_i)}, Second point mask sum: {np.sum(mask_j)}")
                    continue

                line = np.linspace(proj_i, proj_j, 12)
                if np.linalg.norm(line[0] - line[-1]) < 50:
                    # if debug_visualize:
                    #     title = f"Too short line, class i: {class_i}, class j: {class_j}"
                    #     # plot the projected points and the line connecting them on top of the segmented image
                    #     plt.imshow(gest_seg_np)
                    #     plt.imshow(mask_i, alpha=0.4)
                    #     plt.imshow(mask_j, alpha=0.4)
                    #     plt.plot([proj_i[0], proj_j[0]], [proj_i[1], proj_j[1]], c='black', lw=2)
                    #     plt.title(title)
                    #     plt.show()
                    continue

                observed_ims += 1

                # ipdb.set_trace()
                # plot the line and the projected points on top of the segmented image and show

                # plt.imshow(gest_seg_np)
                # plt.scatter(proj_i[0], proj_i[1], c='red', s=50)
                # plt.scatter(proj_j[0], proj_j[1], c='red', s=50)
                # plt.plot(line[:,0], line[:,1], c='black', lw=2)
                # plt.show()

                support_pts = 0
                # The segmentation can be occluded by other things and that would need to be accounted for
                for pt in line[1:-1]:
                    if appropriate_line_close(pt, gest_seg_np, class_i,class_j, patch_size = 5):
                        support_pts += 1
                # print("Support pts: ", support_pts)
                if support_pts >= 6:
                    support_ims += 1


                if debug_visualize  and (class_i == "apex" and class_j == "apex"):
                    title = f"Support points: {support_pts}, decision: {support_pts >= 5}, class i: {class_i}, class j: {class_j}"
                    # plot the projected points and the line connecting them on top of the segmented image
                    fig = plt.figure(figsize=(12,12))
                    plt.imshow(gest_seg_np)
                    plt.scatter(proj_i[0], proj_i[1], c='black', s=25, marker='x')
                    plt.scatter(proj_j[0], proj_j[1], c='black', s=25, marker='x')
                    plt.plot([proj_i[0], proj_j[0]], [proj_i[1], proj_j[1]], c='black', lw=2)
                    plt.title(title)
                    plt.show()
                    # plt.savefig(f"data/visuals_new/debug_edges/house_{house_number}_line_{i}_{j}_image_{k}.png")
                    # plt.close()

            # print("Support ims: ", support_ims)
            # print("Observed ims: ", observed_ims)

            if observed_ims == 0:
                continue
            
            if support_ims/observed_ims > 0.5:
                edges.append([i,j])
                vertex_edge_count[i] += 1
                vertex_edge_count[j] += 1
    
    return edges, vertex_edge_count


def compute_min_dists_to_gt(perdicted_points, gt_points):

    if not isinstance(perdicted_points, np.ndarray):
        perdicted_points = np.array(perdicted_points)
    
    if not isinstance(gt_points, np.ndarray):
        gt_points = np.array(gt_points)

    pwise_dists = np.linalg.norm(perdicted_points[:,None] - gt_points[None], axis = -1)
    min_dists_pred_to_gt = np.min(pwise_dists, axis = 1)
    min_dists_gt_to_pred = np.min(pwise_dists, axis = 0)

    return min_dists_pred_to_gt, min_dists_gt_to_pred


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

def get_scale_from_sfm_points(monocular_depth, sfm_points, K, R, t):

    projected_depth = np.zeros_like(monocular_depth)
    projection_matrix = np.dot(K, np.hstack((R, t.reshape(3,1))))
    sfm_points_h = np.hstack((sfm_points, np.ones((sfm_points.shape[0], 1))))
    projected_pts = np.rint(np.dot(projection_matrix, sfm_points_h.T).T)
    
    projected_pts[:,:2] /= projected_pts[:,2].reshape(-1,1)

    x = projected_pts[:,0].astype(np.int32)
    y = projected_pts[:,1].astype(np.int32)
    z = projected_pts[:,2]

    # print(z.min(), z.max())
    # use only the 25 percent closest points
    max_z = np.percentile(z, 30)
    mask = (x >= 0) & (x < monocular_depth.shape[1]) & (y >= 0) & (y < monocular_depth.shape[0]) & (z < max_z)

    x = x[mask]
    y = y[mask]
    z = z[mask]

    # sort by z values so that the 
    idx = np.argsort(z)
    idx = idx[np.unique(np.ravel_multi_index((y,x), (monocular_depth.shape[0],monocular_depth.shape[1])),return_index=True)[1]]
    
    x = x[idx]
    y = y[idx]

    projected_depth[y,x] = z[idx]

    # Get the median value of the scale factor that aligns the monocular depth to the projected depth
    mask = projected_depth > 0
    scale = np.median(monocular_depth[mask]/projected_depth[mask])
    scale = 1/scale

    return scale, max_z

    