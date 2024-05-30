import open3d as o3d
import numpy as np

def get_triangulated_pts_o3d_pc(triangulated_corners, triangulated_corner_classes, gestalt_color_mapping):
    triangulated_pts_o3d = o3d.geometry.PointCloud()
    triangulated_pts_o3d.points = o3d.utility.Vector3dVector(triangulated_corners)
    colors = np.zeros_like(triangulated_corners)
    apex_inds = np.array([v == 'apex' for v in triangulated_corner_classes]).astype(bool)
    eave_inds = np.array([v == 'eave_end_point' for v in triangulated_corner_classes]).astype(bool)
    flashing_ends_inds = np.array([v == 'flashing_end_point' for v in triangulated_corner_classes]).astype(bool)

    colors[apex_inds] = np.array(gestalt_color_mapping['apex'])
    colors[eave_inds] = np.array(gestalt_color_mapping['eave_end_point'])
    colors[flashing_ends_inds] = np.array(gestalt_color_mapping['flashing_end_point'])
    triangulated_pts_o3d.colors = o3d.utility.Vector3dVector(colors/255.0)
    # ipdb.set_trace()
    return triangulated_pts_o3d

def get_wireframe_o3d(wf_vertices, wf_edges):
    o3d_gt_lines = o3d.geometry.LineSet()
    o3d_gt_lines.points = o3d.utility.Vector3dVector(np.array(wf_vertices))
    o3d_gt_lines.colors = o3d.utility.Vector3dVector(np.array([[0,0,0]]*len(wf_vertices)))
    o3d_gt_lines.lines = o3d.utility.Vector2iVector(np.array(wf_edges))
    return o3d_gt_lines

def get_open3d_point_cloud(triangulated_points, vertex_types):
    """
    Get the open3d point cloud from the vertices
    :param vertices: vertices of the gestalt
    :return: open3d point cloud
    """
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    
    if not isinstance(triangulated_points, np.ndarray):
        triangulated_points = np.array(triangulated_points)
    
    colors = np.zeros_like(triangulated_points)
    apex_inds = np.array([v == 'apex' for v in vertex_types]).astype(bool)
    eave_inds = np.array([v == 'eave_end_point' for v in vertex_types]).astype(bool)
    colors[apex_inds] = np.array(gestalt_color_mapping['apex'])
    colors[eave_inds] = np.array(gestalt_color_mapping['eave_end_point'])

    pcd.points = o3d.utility.Vector3dVector(triangulated_points)
    # print(colors)
    pcd.colors = o3d.utility.Vector3dVector(colors/255.0)
    
    return pcd

def get_open3d_lines(verts, edges):
    import open3d as o3d
    o3d_lineset = o3d.geometry.LineSet()
    o3d_lineset.points = o3d.utility.Vector3dVector(np.array(verts).reshape(-1,3))
    o3d_lineset.lines = o3d.utility.Vector2iVector(np.array(edges).reshape(-1,2))
    return o3d_lineset
