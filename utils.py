import numpy as np
# import torch
import cv2
from hoho.color_mappings import gestalt_color_mapping
import matplotlib.pyplot as plt
# import ipdb

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

def triangulate_pair(Ks, Rs, ts, vertices, verbose = False):

    triangulated_points = []
    vertex_types = []
    proj_mat0 = np.dot(Ks[0], np.hstack((Rs[0], ts[0].reshape(3,1))))
    proj_mat1 = np.dot(Ks[1], np.hstack((Rs[1], ts[1].reshape(3,1))))
    for v1 in vertices[0]:
        for v2 in vertices[1]:
            if v1['type'] == v2['type']:
                
                X,reprojection_errs  = triangulate_multiview_algebraic_error([proj_mat0, proj_mat1], [v1['xy'], v2['xy']])
                if np.sum(reprojection_errs) < 10:
                    if verbose:
                        d1 = np.dot(Rs[0][2, :], X) + ts[0][2]
                        d2 = np.dot(Rs[1][2, :], X) + ts[1][2]
                        print(f"Depths in cameras {d1} and {d2}")
                        print(f"Reprojection errors: {reprojection_errs[0]} and {reprojection_errs[1]}")
                    
                    triangulated_points += [X]                
                    vertex_types += [v1['type']]

    return triangulated_points, vertex_types

def triangulate_from_viewpoints(Ks, Rs, ts, vertices):
    assert len(Ks) == len(Rs) == len(ts) == len(vertices)
    triangulated_points = []
    vertex_types = []
    num_views = len(Ks)
    for i in range(num_views):
        for j in range(i+1, num_views):
            Ks_ = [Ks[i], Ks[j]]
            Rs_ = [Rs[i], Rs[j]]
            ts_ = [ts[i], ts[j]]
            triangulated_points_, vertex_types_ = triangulate_pair(Ks_, Rs_, ts_, [vertices[i], vertices[j]])
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


def get_triangulated_corners(gestalt_segmented_images, depth_images, Ks, Rs, ts):
    
    vertices_all = []
    for segim in gestalt_segmented_images:
        gest_seg_np = np.array(segim)
        vertices = get_vertices_from_gestalt(gest_seg_np)
        # plot_centroids(vertices, gest_seg_np)
        vertices_all.append(vertices)

    # Need to add robutness here, some points are too far off.
    triangulated_vertices, vertex_types = triangulate_from_viewpoints(Ks, Rs, ts, vertices_all)
    return triangulated_vertices, vertex_types


def appropriate_line_close(pt, gest_seg_np, class_i, class_j, patch_size = 10, thresh = 10):
    # If class i and class j are both apex, then we are looking for a ridge
    # If class i is apex and class j is eave, then we are looking for a rake or valley
    # If class i is eave and class j is apex, then we are looking for a rake or valley
    # If class i and class j are both eave, then we are looking for a eave
    # If class i is flashing_end_point and class j is apex, then we are looking for rake
    # If class i is flashing_end_point and class j is eave, then we are looking for rake
    # Not dealing with other cases for now
    ridge_color = np.array(gestalt_color_mapping['ridge'])
    rake_color = np.array(gestalt_color_mapping['rake'])
    eave_color = np.array(gestalt_color_mapping['eave'])
    valley_color = np.array(gestalt_color_mapping['valley'])
    mask = None
    # get a window around the point
    window = gest_seg_np[int(pt[1])-2:int(pt[1])+patch_size//2, int(pt[0])-2:int(pt[0])+patch_size//2]
    if class_i == 'apex' and class_j == 'apex':
        mask = cv2.inRange(window, ridge_color-thresh/2, ridge_color+thresh/2)
    elif class_i == 'apex' and class_j == 'eave_end_point':
        mask = cv2.inRange(window, rake_color-thresh/2, rake_color+thresh/2)
    elif class_i == 'eave_end_point' and class_j == 'apex':
        mask = cv2.inRange(window, rake_color-thresh/2, rake_color+thresh/2)
    elif class_i == 'eave_end_point' and class_j == 'eave_end_point':
        mask = cv2.inRange(window, eave_color-thresh/2, eave_color+thresh/2)
    elif class_i == 'flashing_end_point' and class_j == 'apex':
        mask = cv2.inRange(window, rake_color-thresh/2, rake_color+thresh/2)
    elif class_i == 'flashing_end_point' and class_j == 'eave_end_point':
        mask = cv2.inRange(window, rake_color-thresh/2, rake_color+thresh/2)
    
    if (mask is not None) and (np.sum(mask) > 1):
        return True
    
    return False

def get_edges_with_support(points_3d_coords, points_3d_classes, gestalt_segmented_images, Ks, Rs, ts):
    # For each edge, find the supporting points
    edges = []
    for i in range(len(points_3d_coords)):
        for j in range(i+1, len(points_3d_coords)):
            
            support_ims = 0
            observed_ims = 0

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
                class_i = points_3d_classes[i]
                class_j = points_3d_classes[j]

                # First check if the projections are in the image
                if proj_i[0] < 0 or proj_i[0] >= gest_seg_np.shape[1] or proj_i[1] < 0 or proj_i[1] >= gest_seg_np.shape[0]:
                    continue
                if proj_j[0] < 0 or proj_j[0] >= gest_seg_np.shape[1] or proj_j[1] < 0 or proj_j[1] >= gest_seg_np.shape[0]:
                    continue

                # Get the 3x3 region around the projections, apply the mask of the class color +- threshold and check the sum of the mask
                # If the sum is greater than 5, then the point is in the region of the class
                try:
                    color_i = np.array(gestalt_color_mapping[class_i])
                    
                    min_y_i = max(int(proj_i[1]) - 1 ,0)
                    max_y_i = min(int(proj_i[1])+2, gest_seg_np.shape[0])

                    min_x_i = max(int(proj_i[0]) - 1, 0)
                    max_x_i = min(int(proj_i[0])+2, gest_seg_np.shape[1])

                    window_i = gest_seg_np[min_y_i:max_y_i, min_x_i:max_x_i]
                    mask_i = cv2.inRange(window_i, color_i-5, color_i+5)

                    color_j = np.array(gestalt_color_mapping[class_j])
                    min_y_j = max(int(proj_j[1]) - 1 ,0)
                    max_y_j = min(int(proj_j[1])+2, gest_seg_np.shape[0])

                    min_x_j = max(int(proj_j[0]) - 1, 0)
                    max_x_j = min(int(proj_j[0])+2, gest_seg_np.shape[1])

                    window_j = gest_seg_np[min_y_j:max_y_j, min_x_j:max_x_j]
                    mask_j = cv2.inRange(window_j, color_j-5, color_j+5)
                except:
                    ipdb.set_trace()

                if np.sum(mask_i) > 5 and np.sum(mask_j) < 5:
                    continue

                line = np.linspace(proj_i, proj_j, 12)
                if np.linalg.norm(line[0] - line[-1]) < 30:
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
                for pt in line[1:-1]:
                    if appropriate_line_close(pt, gest_seg_np, class_i,class_j, patch_size = 5):
                        support_pts += 1
                # print("Support pts: ", support_pts)
                if support_pts >= 6:
                    support_ims += 1

                

            # print("Support ims: ", support_ims)
            # print("Observed ims: ", observed_ims)

            if observed_ims == 0:
                continue
            
            if support_ims/observed_ims > 0.5:
                edges.append([i,j])    
    
    return edges
