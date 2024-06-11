import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import hoho
from pathlib import Path
from utils import get_vertices_from_gestalt, compute_min_dists_to_gt, get_monocular_depths_at, get_scale_from_sfm_points, get_edges_with_support, check_edge_2d, compute_WED, compute_distances_to_line, PlaneModel
from utils_new import triangulate_from_viewpoints
from o3d_utils import get_triangulated_pts_o3d_pc, visualize_3d_line_debug, process_sfm_pc
import ipdb
from utils import process_points
from skimage.measure import LineModelND, ransac
from hoho import vis
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# from hoho import compute_WED
class HouseData():

    def __init__(self, sample):
        
        self.house_key = sample['__key__']
        self.gestalt_images = sample['gestalt']
        self.monocular_depths = sample['depthcm']
        self.monocular_depth_scale = 0.32

        self.Ks = sample['K']
        self.Rs = sample['R']
        self.ts = sample['t']

        self.gt_wf_vertices = sample['wf_vertices']
        self.gt_wf_edges = sample['wf_edges']
        self.pred_wf_vertices = None
        self.pred_wf_edges = None

        points3d = sample['points3d']
        points3d = np.array([points3d[point3d].xyz for point3d in points3d])
        too_far = points3d[:, 2] > 800
        # print(points3d[~too_far].shape)
        self.sfm_points = points3d[~too_far]

        # print(len(self.gestalt_images), len(self.monocular_depths), len(self.Ks), len(self.Rs), len(self.ts))

    def get_2d_corners(self):
        vertices_2d = []

        for i,segim in enumerate(self.gestalt_images):
            gest_seg_np = np.array(segim)
            vertices = get_vertices_from_gestalt(gest_seg_np)
            # for vertex in vertices:
            #     vertex['image_id'] = i
            #     vertices_2d.append(vertex)
            vertices_2d.append(vertices)
            # print(len(vertices))

        self.vertices_2d = vertices_2d

    def triangulate_all_2d_corner_pairs(self):
        """
        Triangulate all 2D corner pairs in the house
        """
        triangulated_corners, vertex_types, image_vertex_inds = triangulate_from_viewpoints(Ks = self.Ks, 
                                                                          Rs = self.Rs, 
                                                                          ts = self.ts, 
                                                                          vertices = self.vertices_2d, 
                                                                          segmented_images = self.gestalt_images,
                                                                          debug_visualize = False,
                                                                          gt_lines_o3d = None,
                                                                          house_pts = None, 
                                                                          dist_thresh_house_pts = None)

        for i,_ in enumerate(triangulated_corners):
            for k in range(2):
                assoc_vertex = (image_vertex_inds[i][k], image_vertex_inds[i][2][k])
                
                if 'tri_corner_inds' in self.vertices_2d[assoc_vertex[0]][assoc_vertex[1]]:
                    self.vertices_2d[assoc_vertex[0]][assoc_vertex[1]]['tri_corner_inds'] += [i]
                    self.vertices_2d[assoc_vertex[0]][assoc_vertex[1]]['tri_assoc_2d'] += [(image_vertex_inds[i][1-k], image_vertex_inds[i][2][1-k])]
                else:
                    self.vertices_2d[assoc_vertex[0]][assoc_vertex[1]]['tri_corner_inds'] = [i]
                    self.vertices_2d[assoc_vertex[0]][assoc_vertex[1]]['tri_assoc_2d'] = [(image_vertex_inds[i][1-k], image_vertex_inds[i][2][1-k])]
        
        min_dists_pred_to_gt, min_dists_gt_to_pred = compute_min_dists_to_gt(triangulated_corners, self.gt_wf_vertices)
        
        self.triangulated_corners = [{"xyz" : xyz, "type" : vertex_type} for xyz, vertex_type in zip(triangulated_corners, vertex_types)]
         
        # print(np.mean(min_dists_pred_to_gt), np.mean(min_dists_gt_to_pred))

    
    def get_all_corners_using_monocular_depths(self, dist_thresh_house_pts_monocular = None, visualize = False):

        assert self.vertices_2d is not None
        monocular_est_corners = []
        for i,vertices_set in enumerate(self.vertices_2d):
            positions = np.array([vert['xy'] for vert in vertices_set]).astype(np.int32)
            monocular_depth_np = np.array(self.monocular_depths[i])
            scale, max_z = get_scale_from_sfm_points(monocular_depth_np, self.sfm_points, self.Ks[i], self.Rs[i], self.ts[i])
            scale = min(scale, 1.0)
            monocular_est_corners_, mask = get_monocular_depths_at(monocular_depth_np, self.Ks[i], self.Rs[i], self.ts[i], positions, scale = scale, max_z = 2*max_z)
            
            if dist_thresh_house_pts_monocular is not None:
                dists_corners_house_pts = np.linalg.norm(self.house_pts[:, np.newaxis, :] - monocular_est_corners_[np.newaxis, :, :], axis = 2)
                min_dists = np.min(dists_corners_house_pts, axis = 0)

            for i,vertex in enumerate(vertices_set):
                if mask[i]:
                    if dist_thresh_house_pts_monocular  is not None:
                        if min_dists[i] < dist_thresh_house_pts_monocular[vertex['type']]:
                            vertex['monocular_corner'] = monocular_est_corners_[i]
                        else:
                            vertex['monocular_corner'] = None
                    else:
                        vertex['monocular_corner'] = monocular_est_corners_[i]
                else:
                    vertex['monocular_corner'] = None

            monocular_est_corners += [monocular_est_corners_[i] for i in range(len(vertices_set)) if mask[i]]
                    
            if visualize:

                o3d_mnocular_depth_pts = o3d.geometry.PointCloud()
                o3d_mnocular_depth_pts.points = o3d.utility.Vector3dVector(monocular_est_corners_)
                colors = np.zeros_like(monocular_est_corners_)
                colors[mask] = np.array([0, 255, 0])
                o3d_mnocular_depth_pts.colors = o3d.utility.Vector3dVector(colors)

                o3d_gt_wf = o3d.geometry.LineSet()
                o3d_gt_wf.points = o3d.utility.Vector3dVector(np.array(self.gt_wf_vertices))
                o3d_gt_wf.lines = o3d.utility.Vector2iVector(np.array(self.gt_wf_edges))

                o3d.visualization.draw_geometries([o3d_mnocular_depth_pts, o3d_gt_wf])

        self.monocular_est_corners = monocular_est_corners
        
    def merge_triangulated_monocular_corners(self, merge_neighbors_final = True):
        
        # The monocular depth based points lie on the ray from the camera center to the 2D corner, the triangulated corners should lie close to this ray
        # First we need to associate multiple triangulated corner points with the same 2D point to it if we were to take the approach of selecting a single point for every 2D corner 

        # Another could be to classify based using the information of the sfm points and the vertex class.
        
        # Iterate over all the 2d vertices,
        # visited = {(i,j): False for i in range(len(self.vertices_2d)) for j in range(len(self.vertices_2d[i]))}
        # for i, vertex_set in enumerate(self.vertices_2d):
        #     for j, vertex in enumerate(vertex_set):
        #         if visited[(i,j)]:
        #             continue
        #         if 'tri_corner_inds' in vertex:
                    
        # Create an open3d point cloud of the triangulated corners and the monocular corners
    
        triangulated_corners = [corner['xyz'] for corner in self.triangulated_corners]
        triangulated_corner_classes = [corner['type'] for corner in self.triangulated_corners]
        
        # # ipdb.set_trace()    
        # o3d_triangulated_pts = get_triangulated_pts_o3d_pc(triangulated_corners, triangulated_corner_classes)
        
        # monocular_corners = [corner for corner in self.monocular_est_corners if corner is not None]
        # monocular_corner_colors = np.array([[0, 255, 0]]*len(monocular_corners))
        # o3d_monocular_pts = o3d.geometry.PointCloud()
        # o3d_monocular_pts.points = o3d.utility.Vector3dVector(monocular_corners)
        # o3d_monocular_pts.colors = o3d.utility.Vector3dVector(monocular_corner_colors)
        
        # gt_house_wf = o3d.geometry.LineSet()
        # gt_house_wf.points = o3d.utility.Vector3dVector(np.array(self.gt_wf_vertices))
        # gt_house_wf.lines = o3d.utility.Vector2iVector(np.array(self.gt_wf_edges))
        
        # o3d.visualization.draw_geometries([o3d_triangulated_pts, o3d_monocular_pts, gt_house_wf])
        # save all three
        # o3d.io.write_point_cloud(f"triangulated_corners_{self.house_key}.ply", o3d_triangulated_pts)
        # o3d.io.write_point_cloud(f"monocular_corners_{self.house_key}.ply", o3d_monocular_pts)
        # o3d.io.write_line_set(f"gt_wf_{self.house_key}.ply", gt_house_wf)
        
        merged_pts = []
        merged_pts_classes = []
        outliers_triangulated = {i: False for i in range(len(triangulated_corners))}
        for i, vertex_set in enumerate(self.vertices_2d):
            for j, vertex in enumerate(vertex_set):
                if 'tri_corner_inds' in vertex:
                    if len(vertex['tri_corner_inds']) > 1:

                        if vertex['monocular_corner'] is not None:
                            min_ind = np.argmin([np.linalg.norm(triangulated_corners[tri_ind] - vertex['monocular_corner']) for tri_ind in vertex['tri_corner_inds']])
                            merged_pts.append(triangulated_corners[vertex['tri_corner_inds'][min_ind]])
                            merged_pts_classes.append(vertex['type'])

                            # Mark the other triangulated points as outliers
                            for tri_ind in vertex['tri_corner_inds']:
                                if tri_ind != vertex['tri_corner_inds'][min_ind]:
                                    # outliers_triangulated[tri_ind] = True
                                    continue

                            # Keep the one closest to monocular depth
                            # Visualize all positions
                            # o3d_triangulated = o3d.geometry.PointCloud()
                            # o3d_triangulated.points = o3d.utility.Vector3dVector([triangulated_corners[tri_ind] for tri_ind in vertex['tri_corner_inds']])
                            # o3d_triangulated.colors = o3d.utility.Vector3dVector(np.array([[255, 0, 0]]*len(vertex['tri_corner_inds']))/255.0)

                            # o3d_monocular = o3d.geometry.PointCloud()
                            # o3d_monocular.points = o3d.utility.Vector3dVector([vertex['monocular_corner']]*len(vertex['tri_corner_inds']))
                            # o3d_monocular.colors = o3d.utility.Vector3dVector(np.array([[0, 255, 0]]*len(vertex['tri_corner_inds']))/255.0)

                            # o3d.visualization.draw_geometries([o3d_triangulated, o3d_monocular, gt_house_wf])
                        else:
                            # keep the one with the minimum norm
                            min_ind = np.argmin([np.linalg.norm(triangulated_corners[tri_ind]) for tri_ind in vertex['tri_corner_inds']])
                            merged_pts.append(triangulated_corners[vertex['tri_corner_inds'][min_ind]])
                            merged_pts_classes.append(vertex['type'])

                            for tri_ind in vertex['tri_corner_inds']:
                                if tri_ind != vertex['tri_corner_inds'][min_ind]:
                                    # outliers_triangulated[tri_ind] = True
                                    continue
        
        for i, vertex_set in enumerate(self.vertices_2d):
            for j, vertex in enumerate(vertex_set):
                if 'tri_corner_inds' in vertex:

                    if len(vertex['tri_corner_inds']) == 1:
                        if vertex['monocular_corner'] is not None:
                            # append triangulated if the distance from monocular depth is < 500
                            if (np.linalg.norm(triangulated_corners[vertex['tri_corner_inds'][0]] - vertex['monocular_corner']) < 500) and not outliers_triangulated[vertex['tri_corner_inds'][0]]:
                                merged_pts.append(triangulated_corners[vertex['tri_corner_inds'][0]])
                                merged_pts_classes.append(vertex['type'])
                        else:
                            if not outliers_triangulated[vertex['tri_corner_inds'][0]]:
                                merged_pts.append(triangulated_corners[vertex['tri_corner_inds'][0]])
                                merged_pts_classes.append(vertex['type'])
                else:
                    if vertex['monocular_corner'] is not None:
                        merged_pts.append(vertex['monocular_corner'])
                        merged_pts_classes.append(vertex['type'])
        
        merged_pts_o3d = get_triangulated_pts_o3d_pc(merged_pts, merged_pts_classes)
        # o3d.visualization.draw_geometries([merged_pts_o3d, gt_house_wf])
        
        if merge_neighbors_final:
            merged_pts, merged_pts_classes = process_points(merged_pts, merged_pts_classes, merge = True, merge_threshold = 50, remove = False, append = False)
        
        self.pred_wf_vertices = merged_pts
        self.pred_wf_vertices_classes = merged_pts_classes
    
    
    def get_edges(self, method = "handcrafted", visualize = False):
        if method == "no_edges":
            self.pred_wf_edges = []
        
        elif method == "handcrafted":
            self.pred_wf_edges, _ = get_edges_with_support(self.pred_wf_vertices, self.pred_wf_vertices_classes, 
                                                        self.gestalt_images,
                                                        self.Ks, self.Rs, self.ts,
                                                        horizontal_components = self.horizontal_components,
                                                        vertical_component = self.vertical_component,
                                                        gt_wireframe = [self.gt_wf_vertices, self.gt_wf_edges],
                                                        debug_visualize = False, house_number = "house")
            
            
            # self.pred_wf_edges = []
            # Visualize the predicted and ground truth wireframes
            print(f"House key: {self.house_key}")
            print("Num vertices in predicted wireframe: ", len(self.pred_wf_vertices))
            print("Num edges in predicted wireframe: ", len(self.pred_wf_edges))
            
            if visualize:
                o3d_gt_wf = o3d.geometry.LineSet()
                o3d_gt_wf.points = o3d.utility.Vector3dVector(np.array(self.gt_wf_vertices))
                o3d_gt_wf.lines = o3d.utility.Vector2iVector(np.array(self.gt_wf_edges))

                # o3d_pred_wf = o3d.geometry.LineSet()
                # o3d_pred_wf.points = o3d.utility.Vector3dVector(np.array(self.pred_wf_vertices))
                # o3d_pred_wf.lines = o3d.utility.Vector2iVector(np.array(self.pred_wf_edges))
                # if len(self.pred_wf_edges) > 0:
                #     o3d_pred_wf.colors = o3d.utility.Vector3dVector(np.array([[255.0, 0, 0]]*len(self.pred_wf_edges))/255.0)

                o3d_pred_points = get_triangulated_pts_o3d_pc(self.pred_wf_vertices, self.pred_wf_vertices_classes)

                o3d_major_directions = o3d.geometry.LineSet()
                origin = np.array([0, 0, 0])
                end_points = origin + 1000*np.array([[0,0,1], [0,1,0], [1,0,0]])
                all_points = np.vstack([origin, end_points])
                o3d_major_directions.points = o3d.utility.Vector3dVector(all_points)
                o3d_major_directions.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
                # color the major directions r g b
                o3d_major_directions.colors = o3d.utility.Vector3dVector(np.array([[255.0, 0, 0], [0, 255.0, 0], [0, 0, 255.0]])/255.0)

                sfm_points = o3d.geometry.PointCloud()
                sfm_points.points = o3d.utility.Vector3dVector(self.sfm_points)
                sfm_points.paint_uniform_color([0.5, 0.5, 0.5])

                o3d.visualization.draw_geometries([o3d_gt_wf, o3d_pred_points, o3d_major_directions, sfm_points, o3d_major_directions])

        
        elif method == 'new_hc':
            
            self.pred_wf_edges = []
            # First lets only connect the eave end points to other eave end points

            vertices = np.array(self.pred_wf_vertices)
            vertex_classes = np.array(self.pred_wf_vertices_classes)
            eave_end_points_inds = np.where(vertex_classes == 'eave_end_point')[0]
            flashing_end_points_inds = np.where(vertex_classes == 'flashing_end_point')[0]
            
            for i, ind1 in enumerate(eave_end_points_inds):
                # if self.pred_verts_num_close_sfm_pts[ind1] < 5:
                #     continue
                for j, ind2 in enumerate(eave_end_points_inds):
                    # if self.pred_verts_num_close_sfm_pts[ind2] < 5:
                    #     continue
                    if i < j:
                        
                        # print(self.pred_verts_num_close_sfm_pts[ind1])
                        # print(self.pred_verts_num_close_sfm_pts[ind2])

                        line_3d = vertices[ind1] - vertices[ind2]
                        line_dir_3d = line_3d/np.linalg.norm(line_3d)

                        if np.abs(line_dir_3d[2]) > 0.1:
                            continue
                        
                        hor_align = np.abs(np.dot((self.horizontal_components).reshape(2,3), line_dir_3d.reshape(3,1)))
                        if np.max(hor_align) < 0.98:
                            continue

                        if np.max(np.abs(line_dir_3d)) < 0.1:
                            continue
                        
                        # Check the alignment with the vertical component
                        # print(f"Line direction: {line_dir_3d}")
                        
                        # ipdb.set_trace()
                        # visualize_3d_line_debug(self.sfm_points, self.gt_wf_vertices, self.gt_wf_edges, 
                        #                         self.pred_wf_vertices, self.pred_wf_vertices_classes, 
                        #                         np.array([vertices[ind1], vertices[ind2]]), self.horizontal_components)
                        
                        decision_2d = check_edge_2d(self.gestalt_images, self.Ks, self.Rs, self.ts, 
                                                    vertices[ind1], vertices[ind2], vertex_classes[ind1], vertex_classes[ind2],
                                                    debug_visualize = False, house_number = f"{self.house_key}_{ind1}_{ind2}")

                        if decision_2d:
                            # print("Edge accepted")
                            self.pred_wf_edges += [(ind1, ind2)]
                            # print("Edge rejected")

            # Each eave can only be connected to one other eave_end_point
            # if there are multiple connections, only keep the one with the minimum distance

            keep_ = "none"
            for i, ind1 in enumerate(eave_end_points_inds):
                # check the number of connections
                connected_inds = [edge[1] for edge in self.pred_wf_edges if edge[0] == ind1] + [edge[0] for edge in self.pred_wf_edges if edge[1] == ind1]
                # visualize all connections, stack the point with all connected points

                if len(connected_inds) > 1:
                    # keep the one with the minimum distance

                    if keep_ == "most_aligned":
                        connected_points = [vertices[ind1]] + [vertices[ind2] for ind2 in connected_inds]
                        connected_points = np.vstack(connected_points)
                        # visualize_3d_line_debug(self.sfm_points, self.gt_wf_vertices, self.gt_wf_edges, 
                        #                             self.pred_wf_vertices, self.pred_wf_vertices_classes, 
                        #                             connected_points, self.horizontal_components)
                        
                        # keep at most one line along each of the horizontal directions
                        line_dirs_3d = np.array([vertices[ind2] - vertices[ind1] for ind2 in connected_inds])
                        line_dirs_3d = line_dirs_3d/np.linalg.norm(line_dirs_3d, axis = 1, keepdims=True)
                        dir1_inds = np.where(np.abs(np.dot(self.horizontal_components[0], line_dirs_3d.T)) > 0.98)[0]
                        dir2_inds = np.where(np.abs(np.dot(self.horizontal_components[1], line_dirs_3d.T)) > 0.98)[0]
                        
                        if len(dir1_inds) > 1:
                            # dists_dir1 = [np.linalg.norm(vertices[ind1] - vertices[connected_inds[ind2]]) for ind2 in dir1_inds]
                            max_alignment_ind = np.argmax(np.abs(np.dot(self.horizontal_components[0], line_dirs_3d[dir1_inds].T)))
                            # min_ind = np.argmin(dists_dir1)
                            keep = connected_inds[dir1_inds[max_alignment_ind]]
                            # ipdb.set_trace()
                            for ind2 in dir1_inds:
                                if keep != connected_inds[ind2]:
                                    self.pred_wf_edges.remove((ind1, connected_inds[ind2])  if (ind1, connected_inds[ind2]) in self.pred_wf_edges else (connected_inds[ind2], ind1))
                        
                        if len(dir2_inds) > 1:
                            # dists_dir2 = [np.linalg.norm(vertices[ind1] - vertices[connected_inds[ind2]]) for ind2 in dir2_inds]
                            # min_ind = np.argmin(dists_dir2)
                            max_alignment_ind = np.argmax(np.abs(np.dot(self.horizontal_components[1], line_dirs_3d[dir2_inds].T)))
                            # min_ind = np.argmin(dists_dir1)
                            keep = connected_inds[dir2_inds[max_alignment_ind]]
                            # keep = connected_inds[dir2_inds[min_ind]]
                            # ipdb.set_trace()
                            for ind2 in dir2_inds:
                                if keep != connected_inds[ind2]:
                                    self.pred_wf_edges.remove((ind1, connected_inds[ind2]) if (ind1, connected_inds[ind2]) in self.pred_wf_edges else (connected_inds[ind2], ind1))
                    
                    elif keep_ == "none":
                        # remove all edges
                        for ind2 in connected_inds:
                            self.pred_wf_edges.remove((ind1, ind2) if (ind1, ind2) in self.pred_wf_edges else (ind2, ind1))

            # Add the eave and apex edges
            apex_inds = np.where(vertex_classes == 'apex')[0]
            eave_inds = np.where(vertex_classes == 'eave_end_point')[0]

            for apex_ind in apex_inds:
                
                # if self.pred_verts_num_close_sfm_pts[apex_ind] < 5:
                    # continue
                
                for eave_ind in eave_inds:
                    
                    # if self.pred_verts_num_close_sfm_pts[eave_ind] < 5:
                        # continue

                    line_3d = vertices[apex_ind] - vertices[eave_ind]
                    line_dir_3d = line_3d/np.linalg.norm(line_3d)
                    
                    hor_align = np.abs(np.dot((self.horizontal_components).reshape(2,3), line_dir_3d.reshape(3,1)))
                    min_hor_align = np.min(hor_align)
                    if min_hor_align > 0.1:
                        continue

                    # print(f"Line direction: {line_dir_3d}")
                    # print(f"Horizontal alignment: {hor_align}")

                    # visualize_3d_line_debug(self.sfm_points, self.gt_wf_vertices, self.gt_wf_edges, 
                    #                         self.pred_wf_vertices, self.pred_wf_vertices_classes, 
                    #                         np.array([vertices[apex_ind], vertices[eave_ind]]), self.horizontal_components)
                    
                    decision_2d = check_edge_2d(self.gestalt_images, self.Ks, self.Rs, self.ts, 
                                                    vertices[apex_ind], vertices[eave_ind], vertex_classes[apex_ind], vertex_classes[eave_ind],
                                                    debug_visualize = False, house_number = f"{self.house_key}_{ind1}_{ind2}")

                    if decision_2d:
                        # print("Edge accepted")
                        self.pred_wf_edges += [(apex_ind, eave_ind)]
                    # else:
                    #     print("Edge rejected")
            
            for i, ind1 in enumerate(apex_inds):
                connected_inds = [edge[1] for edge in self.pred_wf_edges if edge[0] == ind1] + [edge[0] for edge in self.pred_wf_edges if edge[1] == ind1]
                if len(connected_inds) > 1:
                    line_dirs_3d = np.array([vertices[ind2] - vertices[ind1] for ind2 in connected_inds])
                    line_dirs_3d = line_dirs_3d/np.linalg.norm(line_dirs_3d)
                    # check pairwise alignment
                    for j, ind2 in enumerate(connected_inds):
                        for k, ind3 in enumerate(connected_inds):
                            if j < k:
                                if np.abs(np.dot(line_dirs_3d[j], line_dirs_3d[k])) > 0.98:
                                    # remove the one with the maximum distance
                                    if np.linalg.norm(vertices[ind1] - vertices[ind2]) > np.linalg.norm(vertices[ind1] - vertices[ind3]):
                                        self.pred_wf_edges.remove((ind1, ind2) if (ind1, ind2) in self.pred_wf_edges else (ind2, ind1))
                                    else:
                                        self.pred_wf_edges.remove((ind1, ind3) if (ind1, ind3) in self.pred_wf_edges else (ind3, ind1))
            
            
            for i, ind1 in enumerate(apex_inds):
                # if self.pred_verts_num_close_sfm_pts[ind1] < 5:
                    # continue
                for j, ind2 in enumerate(apex_inds):
                    # if self.pred_verts_num_close_sfm_pts[ind2] < 5:
                        # continue
                    if i < j:
                        line_3d = vertices[ind1] - vertices[ind2]
                        line_dir_3d = line_3d/np.linalg.norm(line_3d)

                        if np.abs(line_dir_3d[2]) > 0.05:
                            continue
                        
                        hor_align = np.abs(np.dot((self.horizontal_components).reshape(2,3), line_dir_3d.reshape(3,1)))
                        if np.max(hor_align) < 0.98:
                            continue
                        
                        decision_2d = check_edge_2d(self.gestalt_images, self.Ks, self.Rs, self.ts, 
                                                    vertices[ind1], vertices[ind2], vertex_classes[ind1], vertex_classes[ind2],
                                                    debug_visualize = False, house_number = f"{self.house_key}_{ind1}_{ind2}")

                        if decision_2d:
                            # print("Edge accepted")
                            self.pred_wf_edges += [(ind1, ind2)]
            
            # Each apex can only be connected to one other apex point
            # if there are multiple connections, only keep the one with the minimum distance
            for i, ind1 in enumerate(apex_inds):
                connected_inds = [edge[1] for edge in self.pred_wf_edges if ((edge[0] == ind1) and (vertex_classes[edge[1]] == "apex"))] + \
                                    [edge[0] for edge in self.pred_wf_edges if ((edge[1] == ind1) and (vertex_classes[edge[0]] == "apex"))]
                
                if len(connected_inds) > 1:
                    # remove all edges
                    for ind2 in connected_inds:
                        self.pred_wf_edges.remove((ind1, ind2) if (ind1, ind2) in self.pred_wf_edges else (ind2, ind1))
            
            
            
            # Eave flashing points connections
            for eave_ind in eave_inds:
                
                # if self.pred_verts_num_close_sfm_pts[apex_ind] < 5:
                    # continue
                
                for flashing_ind in flashing_end_points_inds:
                    
                    # if self.pred_verts_num_close_sfm_pts[eave_ind] < 5:
                        # continue

                    line_3d = vertices[flashing_ind] - vertices[eave_ind]
                    line_dir_3d = line_3d/np.linalg.norm(line_3d)
                    
                    hor_align = np.abs(np.dot((self.horizontal_components).reshape(2,3), line_dir_3d.reshape(3,1)))
                    max_hor_align = np.max(hor_align)
                    if max_hor_align < 0.85:
                        continue
                    
                    decision_2d = check_edge_2d(self.gestalt_images, self.Ks, self.Rs, self.ts, 
                                                    vertices[eave_ind], vertices[flashing_ind], vertex_classes[eave_ind], vertex_classes[flashing_ind],
                                                    debug_visualize = False, house_number = f"{self.house_key}_{ind1}_{ind2}")
                    
                    if decision_2d:
                            # print("Edge accepted")
                            self.pred_wf_edges += [(eave_ind, flashing_ind)]
                            
            for i, ind1 in enumerate(eave_inds):
                connected_inds = [edge[1] for edge in self.pred_wf_edges if ((edge[0] == ind1) and (vertex_classes[edge[1]] == "flashing_end_point"))] + \
                                    [edge[0] for edge in self.pred_wf_edges if ((edge[1] == ind1) and (vertex_classes[edge[0]] == "flashing_end_point"))]
                
                if len(connected_inds) > 1:
                    # remove all edges
                    for ind2 in connected_inds:
                        self.pred_wf_edges.remove((ind1, ind2) if (ind1, ind2) in self.pred_wf_edges else (ind2, ind1))
            
            
            # flashing points connections
            for i,ind1 in enumerate(flashing_end_points_inds):
                
                # if self.pred_verts_num_close_sfm_pts[apex_ind] < 5:
                    # continue
                
                for j,ind2 in enumerate(flashing_end_points_inds):
                    
                    # if self.pred_verts_num_close_sfm_pts[eave_ind] < 5:
                        # continue
                    if i<j:
                        line_3d = vertices[ind1] - vertices[ind2]
                        line_dir_3d = line_3d/np.linalg.norm(line_3d)
                        
                        hor_align = np.abs(np.dot((self.horizontal_components).reshape(2,3), line_dir_3d.reshape(3,1)))
                        max_hor_align = np.max(hor_align)
                        if max_hor_align < 0.99:
                            continue
                        
                        decision_2d = check_edge_2d(self.gestalt_images, self.Ks, self.Rs, self.ts, 
                                                        vertices[ind1], vertices[ind2], vertex_classes[ind1], vertex_classes[ind2],
                                                        debug_visualize = False, house_number = f"{self.house_key}_{ind1}_{ind2}")
                        
                        if decision_2d:
                                # print("Edge accepted")
                                self.pred_wf_edges += [(ind1, ind2)]
                            
            for i, ind1 in enumerate(flashing_end_points_inds):
                connected_inds = [edge[1] for edge in self.pred_wf_edges if ((edge[0] == ind1) and (vertex_classes[edge[1]] == "flashing_end_point"))] + \
                                    [edge[0] for edge in self.pred_wf_edges if ((edge[1] == ind1) and (vertex_classes[edge[0]] == "flashing_end_point"))]
                
                if len(connected_inds) > 1:
                    # remove all edges
                    for ind2 in connected_inds:
                        self.pred_wf_edges.remove((ind1, ind2) if (ind1, ind2) in self.pred_wf_edges else (ind2, ind1))
                       
        else:
            raise NotImplementedError
        
    def compute_metric(self):
        computed_wed = compute_WED(self.pred_wf_vertices, self.pred_wf_edges, self.gt_wf_vertices, self.gt_wf_edges)
        self.wed = computed_wed
        print(f"Computed WED for {self.house_key} is {computed_wed}")


    def get_sfm_pca(self):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca.fit(self.sfm_points)
        self.major_directions = pca.components_
        self.horizontal_components = np.array([pca.components_[i] for i in range(3) if abs(pca.components_[i][2]) < 0.8])
        self.vertical_component = np.array([pca.components_[i] for i in range(3) if abs(pca.components_[i][2]) > 0.8])
        if len(self.horizontal_components) != 2:
            self.horizontal_components = None
        if len(self.vertical_component) == 0:
            self.vertical_component = None
        # ipdb.set_trace()


    def get_lines_from_sfm_points(self, visualize = True):

        # project all points onto the ground plane
        points_horizontal = self.sfm_points[:, :2]
        num_points = 10000
        if points_horizontal.shape[0] > num_points:
            points_horizontal = points_horizontal[np.random.choice(points_horizontal.shape[0], num_points, replace = False)]
        # detect lines in the horizontal plane using RANSAC until we have 2 lines which are almost orthogonal

        # ipdb.set_trace()

        # fit a line to the points
        model = LineModelND()
        model.estimate(points_horizontal)

        model_robust, inliers = ransac(points_horizontal, LineModelND, min_samples=2, residual_threshold=10, max_trials=1000)
        #visualize the inliers using open3d
        line1_dir_2d = model_robust.params[1]
        line1_dir_3d = np.array([model_robust.params[1][0], model_robust.params[1][1], 0])
        line1_dir_3d = line1_dir_3d/np.linalg.norm(line1_dir_3d)

        # The second line should be orthogonal to the first line and have z component 0
        line2_dir_2d = np.array([-line1_dir_2d[1], line1_dir_2d[0]])
        line2_dir_3d = np.array([line2_dir_2d[0], line2_dir_2d[1], 0])
        line2_dir_3d = line2_dir_3d/np.linalg.norm(line2_dir_3d)

        self.horizontal_components = np.array([line1_dir_3d, line2_dir_3d])

        # if visualize:
        # o3d_major_directions = o3d.geometry.LineSet()
        # origin = np.array([0, 0, 0])
        # end_points = origin + 1000*np.array([line1_dir_3d, line2_dir_3d, [0, 0, 1]])
        # all_points = np.vstack([origin, end_points])
        # o3d_major_directions.points = o3d.utility.Vector3dVector(all_points)
        # o3d_major_directions.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
        # # color the major directions r g b
        # o3d_major_directions.colors = o3d.utility.Vector3dVector(np.array([[255.0, 0, 0], [0, 255.0, 0], [0, 0, 255.0]])/255.0)

        # o3d_gt_wf = o3d.geometry.LineSet()
        # o3d_gt_wf.points = o3d.utility.Vector3dVector(np.array(self.gt_wf_vertices))
        # o3d_gt_wf.lines = o3d.utility.Vector2iVector(np.array(self.gt_wf_edges))


        # inlier_points = points_horizontal[inliers]
        # o3d_inliers = o3d.geometry.PointCloud()
        # o3d_inliers.points = o3d.utility.Vector3dVector(np.hstack([inlier_points, np.zeros((inlier_points.shape[0], 1))]))
        # o3d_inliers.paint_uniform_color([0, 0, 1])

        # sfm_points = o3d.geometry.PointCloud()
        # sfm_points.points = o3d.utility.Vector3dVector(self.sfm_points)
        # sfm_points.paint_uniform_color([0.5, 0.5, 0.5])

        # o3d_pred_points = get_triangulated_pts_o3d_pc(self.pred_wf_vertices, self.pred_wf_vertices_classes)

        # o3d.visualization.draw_geometries([sfm_points, o3d_gt_wf, o3d_major_directions, o3d_inliers, o3d_pred_points])

    def get_num_sfm_points_within(self, threshold = 100):
        
        points_query = self.pred_wf_vertices
        sfm_points = self.sfm_points

        all_dists = np.linalg.norm(sfm_points[:, None] - points_query, axis = -1)
        self.pred_verts_num_close_sfm_pts = np.sum(all_dists < threshold, axis = 0).reshape(-1)

    
    def plot_2d(self):
        for i, im in enumerate(self.gestalt_images):
            plt.imshow(im)
            for vertex in self.vertices_2d[i]:
                plt.scatter(vertex['xy'][0], vertex['xy'][1], marker='x', s=30, color='black')
            plt.savefig(f"data/visuals_new/june10_2d/{self.house_key}_{i}.png")
            plt.close()
            
            # plot the depth images as well
            # plt.imshow(self.monocular_depths[i])
            depth_image_pil = vis.visualize_depth(self.monocular_depths[i])
            depth_image_np = np.array(depth_image_pil)
            plt.imshow(depth_image_np)
            for vertex in self.vertices_2d[i]:
                plt.scatter(vertex['xy'][0], vertex['xy'][1], marker='x', s=30, color='black')
            plt.savefig(f"data/visuals_new/june10_2d/{self.house_key}_{i}_depth.png")
            plt.close()
            # highlight the top 30 percentile
            # plt.show()
            
    # def get_sfm_wall_lines(self):
        
    #     points_horizontal = self.sfm_points[:, :2]
    #     # num_points = 10000
    #     # if points_horizontal.shape[0] > num_points:
    #         # points_horizontal = points_horizontal[np.random.choice(points_horizontal.shape[0], num_points, replace = False)]
    #     # detect lines in the horizontal plane using RANSAC until we have 2 lines which are almost orthogonal

    #     # ipdb.set_trace()

    #     # fit a line to the points
    #     model = LineModelND()
    #     model.estimate(points_horizontal)

    #     while inliers < 0.9*points_horizontal.shape[0]:        
    #         model_robust, inliers = ransac(points_horizontal, LineModelND, min_samples=2, residual_threshold=10, max_trials=1000)
        
    #     #visualize the inliers using open3d
    #     line1_dir_2d = model_robust.params[1]
    #     line1_dir_3d = np.array([model_robust.params[1][0], model_robust.params[1][1], 0])
    #     line1_dir_3d = line1_dir_3d/np.linalg.norm(line1_dir_3d)
        
    #     return

    def process_sfm_pc(self):

        self.house_pts = process_sfm_pc(self.sfm_points)
        
    def get_monocular_depths_from_sfm_intersection(self):

        # get the monocular depths at the intersection of the sfm points with the horizontal plane
        
        monocular_est_corners = []
        ray_line_points = []
        ray_line_edges = []

        for i,vertices_set in enumerate(self.vertices_2d):
            
            R = self.Rs[i]
            t = self.ts[i]
            K = self.Ks[i]
            Kinv = np.linalg.inv(K)

            positions = np.array([vert['xy'] for vert in vertices_set]).astype(np.int32)
            cam_center = -np.dot(R.T, t)
            # print(cam_center)
            curr_center_ind = len(ray_line_points)
            # print(curr_center_ind)
            ray_line_points.append(cam_center)

            mask = {j: False for j in range(len(vertices_set))}
            for j, vertex in enumerate(vertices_set):
                
                position = positions[j]

                # Convert the 2D pixel position to homogeneous coordinates
                uv = np.array([position[0], position[1], 1]).reshape(3, 1)

                # Back-project to 3D camera coordinates
                X_cam = np.dot(Kinv, uv)

                # Ensure the direction is correct (from camera center towards the pixel)
                X_ray_cam = X_cam * 2000  # Scale the ray

                # Transform the ray from camera coordinates to world coordinates
                X_ray_w = np.dot(R.T, X_ray_cam).T + cam_center

                # Append the ray end point to ray_line_points
                ray_line_points.append(X_ray_w.flatten())

                # Add the edge from the camera center to the ray end point
                ray_line_edges.append([curr_center_ind, curr_center_ind + j + 1])

                # print(f"Adding edge: {curr_center_ind} {curr_center_ind + j + 1}")

                # find the closest point of the ray to the house_points
                line_dir = X_ray_w - cam_center
                line_dir = line_dir/np.linalg.norm(line_dir)
                house_pt_dists = compute_distances_to_line(self.house_pts, cam_center, line_dir.reshape(3,1))
                five_closest_pts = np.argsort(house_pt_dists)[:5]
                closest_house_pt_ind = five_closest_pts[np.argmin(np.linalg.norm(self.house_pts[five_closest_pts] - cam_center, axis = 1))]
                # ipdb.set_trace()
                closest_house_pt = self.house_pts[closest_house_pt_ind]
                # if np.linalg.norm(closest_house_pt - cam_center) < 2000:
                vertex['monocular_corner_new'] = closest_house_pt
                mask[j] = True
                
            monocular_est_corners += [vertices_set[j]['monocular_corner_new'] for j in range(len(vertices_set)) if mask[j]]
        
        self.monocular_est_corners_new = monocular_est_corners

        # ray_line_points = np.vstack(ray_line_points)

        

        # o3d_ray_line = o3d.geometry.PointCloud()
        # o3d_ray_line.points = o3d.utility.Vector3dVector(ray_line_points)

        # ray_line_edges = np.array(ray_line_edges).astype(np.int32)
        # o3d_ray_line = o3d.geometry.LineSet()
        # o3d_ray_line.points = o3d.utility.Vector3dVector(ray_line_points)
        # o3d_ray_line.lines = o3d.utility.Vector2iVector(ray_line_edges)
        # o3d_ray_line.colors = o3d.utility.Vector3dVector(np.array([[255, 0, 0]]*len(ray_line_edges))/255.0)

        # o3d_house_points = o3d.geometry.PointCloud()
        # o3d_house_points.points = o3d.utility.Vector3dVector(self.house_pts)
        # o3d_house_points.paint_uniform_color([0.5, 0.5, 0.5])

        # o3d.visualization.draw_geometries([o3d_ray_line, o3d_house_points])

        # #     monocular_depth_np = np.array(self.monocular_depths[i])
        # #     scale, max_z = get_scale_from_sfm_points(monocular_depth_np, self.sfm_points, self.Ks[i], self.Rs[i], self.ts[i])
        # #     scale = min(scale, 1.5)
        # #     monocular_est_corners_, mask = get_monocular_depths_at(monocular_depth_np, self.Ks[i], self.Rs[i], self.ts[i], positions, scale = scale, max_z = 2*max_z)
            
        # #     for i,vertex in enumerate(vertices_set):
        # #         vertex['monocular_corner'] = monocular_est_corners_[i] if mask[i] else None
            
        # #     monocular_est_corners += [monocular_est_corners_[i] for i in range(len(vertices_set)) if mask[i]]
                    
        # #     if visualize:

        # #         o3d_mnocular_depth_pts = o3d.geometry.PointCloud()
        # #         o3d_mnocular_depth_pts.points = o3d.utility.Vector3dVector(monocular_est_corners_)
        # #         colors = np.zeros_like(monocular_est_corners_)
        # #         colors[mask] = np.array([0, 255, 0])
        # #         o3d_mnocular_depth_pts.colors = o3d.utility.Vector3dVector(colors)

        # #         o3d_gt_wf = o3d.geometry.LineSet()
        # #         o3d_gt_wf.points = o3d.utility.Vector3dVector(np.array(self.gt_wf_vertices))
        # #         o3d_gt_wf.lines = o3d.utility.Vector2iVector(np.array(self.gt_wf_edges))

        # #         o3d.visualization.draw_geometries([o3d_mnocular_depth_pts, o3d_gt_wf])

        # # self.monocular_est_corners = monocular_est_corners

    def merge_triangulated_monocular_corners_new(self, merge_neighbors_final = True):
    
        triangulated_corners = [corner['xyz'] for corner in self.triangulated_corners]
        # triangulated_corners_arr = np.array(triangulated_corners)
        # dists_to_sfm_pts = np.linalg.norm(triangulated_corners_arr[:, None] - self.sfm_points, axis = -1)
        # keep_inds = np.where(np.min(dists_to_sfm_pts, axis = 1) > 200)[0]
        # triangulated_corners = [corner['xyz'] for i, corner in enumerate(self.triangulated_corners) if i in keep_inds]

        merged_pts = []
        merged_pts_classes = []
        outliers_triangulated = {i: False for i in range(len(triangulated_corners))}
        
        for i, vertex_set in enumerate(self.vertices_2d):
            for j, vertex in enumerate(vertex_set):
                if 'tri_corner_inds' in vertex:
                    if len(vertex['tri_corner_inds']) > 1:

                        if vertex['monocular_corner_new'] is not None:
                            min_ind = np.argmin([np.linalg.norm(triangulated_corners[tri_ind] - vertex['monocular_corner_new']) for tri_ind in vertex['tri_corner_inds']])
                            merged_pts.append(triangulated_corners[vertex['tri_corner_inds'][min_ind]])
                            merged_pts_classes.append(vertex['type'])

                            # Mark the other triangulated points as outliers
                            for tri_ind in vertex['tri_corner_inds']:
                                if tri_ind != vertex['tri_corner_inds'][min_ind]:
                                    # outliers_triangulated[tri_ind] = True
                                    continue

                        else:
                            # keep the one with the minimum norm
                            min_ind = np.argmin([np.linalg.norm(triangulated_corners[tri_ind]) for tri_ind in vertex['tri_corner_inds']])
                            merged_pts.append(triangulated_corners[vertex['tri_corner_inds'][min_ind]])
                            merged_pts_classes.append(vertex['type'])

                            for tri_ind in vertex['tri_corner_inds']:
                                if tri_ind != vertex['tri_corner_inds'][min_ind]:
                                    # outliers_triangulated[tri_ind] = True
                                    continue
        
        for i, vertex_set in enumerate(self.vertices_2d):
            for j, vertex in enumerate(vertex_set):
                if 'tri_corner_inds' in vertex:

                    if len(vertex['tri_corner_inds']) == 1:
                        # if vertex['monocular_corner_new'] is not None:
                        #     # append triangulated if the distance from monocular depth is < 500
                        #     if (np.linalg.norm(triangulated_corners[vertex['tri_corner_inds'][0]] - vertex['monocular_corner_new']) < 500) and not outliers_triangulated[vertex['tri_corner_inds'][0]]:
                        merged_pts.append(triangulated_corners[vertex['tri_corner_inds'][0]])
                        merged_pts_classes.append(vertex['type'])
                        # else:
                        #     if not outliers_triangulated[vertex['tri_corner_inds'][0]]:
                        #         merged_pts.append(triangulated_corners[vertex['tri_corner_inds'][0]])
                        #         merged_pts_classes.append(vertex['type'])
                else:
                    # print("Checking for monocular corner")
                    if (vertex['monocular_corner'] is not None):
                        if vertex['monocular_corner_new'] is not None:
                            if np.linalg.norm(vertex['monocular_corner'] - vertex['monocular_corner_new']) > 500:
                                merged_pts.append(vertex['monocular_corner'])   
                                # merged_pts.append(avg_point)
                                merged_pts_classes.append(vertex['type'])
                                # print("Added monocular corner"
                            else:
                                merged_pts.append(vertex['monocular_corner_new'])
                                merged_pts_classes.append(vertex['type'])
                    else:
                        merged_pts.append(vertex['monocular_corner_new'])
                        merged_pts_classes.append(vertex['type'])
                        # print("Added monocular corner")
        
        merged_pts_o3d = get_triangulated_pts_o3d_pc(merged_pts, merged_pts_classes)
        # o3d.visualization.draw_geometries([merged_pts_o3d, gt_house_wf])
        
        if merge_neighbors_final:
            merged_pts, merged_pts_classes = process_points(merged_pts, merged_pts_classes, merge = True, merge_threshold = 50, remove = False, append = False)
        
        self.pred_wf_vertices = merged_pts
        self.pred_wf_vertices_classes = merged_pts_classes

    def triangulate_all_2d_corner_pairs_new(self):
        """
        Triangulate all 2D corner pairs in the house
        """
        dist_thresh_house_pts ={'eave_end_point': 1000, 'apex': 1000, 'flashing_end_point': 1000}
        triangulated_corners, vertex_types, image_vertex_inds = triangulate_from_viewpoints(Ks = self.Ks, 
                                                                          Rs = self.Rs, 
                                                                          ts = self.ts, 
                                                                          vertices = self.vertices_2d, 
                                                                          segmented_images = self.gestalt_images,
                                                                          debug_visualize = False,
                                                                          gt_lines_o3d = None,
                                                                          house_pts = self.house_pts, 
                                                                          dist_thresh_house_pts = dist_thresh_house_pts)

        for i,_ in enumerate(triangulated_corners):
            for k in range(2):
                assoc_vertex = (image_vertex_inds[i][k], image_vertex_inds[i][2][k])
                
                if 'tri_corner_inds' in self.vertices_2d[assoc_vertex[0]][assoc_vertex[1]]:
                    self.vertices_2d[assoc_vertex[0]][assoc_vertex[1]]['tri_corner_inds'] += [i]
                    self.vertices_2d[assoc_vertex[0]][assoc_vertex[1]]['tri_assoc_2d'] += [(image_vertex_inds[i][1-k], image_vertex_inds[i][2][1-k])]
                else:
                    self.vertices_2d[assoc_vertex[0]][assoc_vertex[1]]['tri_corner_inds'] = [i]
                    self.vertices_2d[assoc_vertex[0]][assoc_vertex[1]]['tri_assoc_2d'] = [(image_vertex_inds[i][1-k], image_vertex_inds[i][2][1-k])]
        
        min_dists_pred_to_gt, min_dists_gt_to_pred = compute_min_dists_to_gt(triangulated_corners, self.gt_wf_vertices)
        
        self.triangulated_corners = [{"xyz" : xyz, "type" : vertex_type} for xyz, vertex_type in zip(triangulated_corners, vertex_types)]

    def merge_triangulated_monocular_corners_keep_all(self, merge_neighbors_final = True):
        
        triangulated_corners = [corner['xyz'] for corner in self.triangulated_corners]

        merged_pts = []
        merged_pts_classes = []

        replace_incorrect = False
        # check if several monocular new point lie on a plane, then use monocular corner
        X = np.array(self.monocular_est_corners_new)
        pwise_dists = np.linalg.norm(X[:, None] - X, axis = -1)
        num_close = np.sum(pwise_dists < 50, axis = 1)
        incorrect_inds = np.where(num_close > 4)[0]
        if np.sum(incorrect_inds) > 5:
            replace_incorrect = True
            # visualize the house pts
            # o3d_house_points = o3d.geometry.PointCloud()
            # o3d_house_points.points = o3d.utility.Vector3dVector(self.house_pts)
            # o3d_house_points.paint_uniform_color([0.5, 0.5, 0.5])

            # o3d_gt_wf = o3d.geometry.LineSet()
            # o3d_gt_wf.points = o3d.utility.Vector3dVector(np.array(self.gt_wf_vertices))
            # o3d_gt_wf.lines = o3d.utility.Vector2iVector(np.array(self.gt_wf_edges))

            # o3d_mc_new = o3d.geometry.PointCloud()
            # o3d_mc_new.points = o3d.utility.Vector3dVector(X)
            # o3d_mc_new.paint_uniform_color([0, 0, 1])

            # o3d_mc = o3d.geometry.PointCloud()
            # o3d_mc.points = o3d.utility.Vector3dVector(np.array(self.monocular_est_corners))
            # o3d_mc.paint_uniform_color([1, 0, 0])

            # o3d.visualization.draw_geometries([o3d_house_points, o3d_gt_wf, o3d_mc_new, o3d_mc])

        for vertex_set in self.vertices_2d:
            for vertex in vertex_set:
                if 'tri_corner_inds' in vertex:
                    for i in vertex['tri_corner_inds']:
                        merged_pts.append(triangulated_corners[i])
                        merged_pts_classes.append(vertex['type'])
                else:
                    if vertex['monocular_corner_new'] is not None:
                        merged_pts.append(vertex['monocular_corner_new'])
                        merged_pts_classes.append(vertex['type'])

                    if replace_incorrect:
                        if vertex['monocular_corner'] is not None:
                            merged_pts.append(vertex['monocular_corner'])
                            merged_pts_classes.append(vertex['type'])

        merge_thresh = 60 if replace_incorrect else 30
        if merge_neighbors_final:
            merged_pts, merged_pts_classes = process_points(merged_pts, merged_pts_classes, merge = True, merge_threshold = merge_thresh, remove = False, append = False)
        
        self.pred_wf_vertices = merged_pts
        self.pred_wf_vertices_classes = merged_pts_classes