import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import hoho
from pathlib import Path
from utils import get_vertices_from_gestalt, compute_min_dists_to_gt, get_monocular_depths_at, get_scale_from_sfm_points, get_edges_with_support
from utils_new import triangulate_from_viewpoints
from o3d_utils import get_triangulated_pts_o3d_pc
import ipdb
from utils import process_points
from hoho import compute_WED

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
                                                                          gt_lines_o3d = None)

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

    
    def get_all_corners_using_monocular_depths(self, visualize = False):

        assert self.vertices_2d is not None
        monocular_est_corners = []
        for i,vertices_set in enumerate(self.vertices_2d):
            positions = np.array([vert['xy'] for vert in vertices_set]).astype(np.int32)
            monocular_depth_np = np.array(self.monocular_depths[i])
            scale, max_z = get_scale_from_sfm_points(monocular_depth_np, self.sfm_points, self.Ks[i], self.Rs[i], self.ts[i])
            scale = min(scale, 0.6)
            monocular_est_corners_, mask = get_monocular_depths_at(monocular_depth_np, self.Ks[i], self.Rs[i], self.ts[i], positions, scale = scale, max_z = 2*max_z)
            
            for i,vertex in enumerate(vertices_set):
                vertex['monocular_corner'] = monocular_est_corners_[i] if mask[i] else None
            
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
                                    outliers_triangulated[tri_ind] = True

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
                                    outliers_triangulated[tri_ind] = True
        
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
            merged_pts, merged_pts_classes = process_points(merged_pts, merged_pts_classes, merge = True, merge_threshold = 20, remove = False, append = False)
        
        self.pred_wf_vertices = merged_pts
        self.pred_wf_vertices_classes = merged_pts_classes
    
    
    def get_edges(self, method = 'handcrafted', visualize = False):
        if method == 'handcrafted':
            self.pred_wf_edges, _ = get_edges_with_support(self.pred_wf_vertices, self.pred_wf_vertices_classes, 
                                                        self.gestalt_images,
                                                        self.Ks, self.Rs, self.ts,
                                                        horizontal_components = self.horizontal_components,
                                                        vertical_component = self.vertical_component,
                                                        gt_wireframe = [self.gt_wf_vertices, self.gt_wf_edges],
                                                        debug_visualize = False, house_number = "house")
            
            
            # Visualize the predicted and ground truth wireframes
            if visualize:
                o3d_gt_wf = o3d.geometry.LineSet()
                o3d_gt_wf.points = o3d.utility.Vector3dVector(np.array(self.gt_wf_vertices))
                o3d_gt_wf.lines = o3d.utility.Vector2iVector(np.array(self.gt_wf_edges))

                o3d_pred_wf = o3d.geometry.LineSet()
                o3d_pred_wf.points = o3d.utility.Vector3dVector(np.array(self.pred_wf_vertices))
                o3d_pred_wf.lines = o3d.utility.Vector2iVector(np.array(self.pred_wf_edges))

                o3d_pred_points = get_triangulated_pts_o3d_pc(self.pred_wf_vertices, self.pred_wf_vertices_classes)

                o3d.visualization.draw_geometries([o3d_gt_wf, o3d_pred_wf, o3d_pred_points])

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

