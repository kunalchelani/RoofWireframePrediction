import open3d as o3d
import numpy as np
from hoho.color_mappings import gestalt_color_mapping
import ipdb

def get_triangulated_pts_o3d_pc(triangulated_corners, triangulated_corner_classes):
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
    pcd = o3d.geometry.PointCloud()
    
    if not isinstance(triangulated_points, np.ndarray):
        triangulated_points = np.array(triangulated_points)
    
    colors = np.zeros_like(triangulated_points)
    apex_inds = np.array([v == 'apex' for v in vertex_types]).astype(bool)
    eave_inds = np.array([v == 'eave_end_point' for v in vertex_types]).astype(bool)
    if sum(apex_inds) > 0:
        colors[apex_inds] = np.array(gestalt_color_mapping['apex'])
    if sum(eave_inds) > 0:
        colors[eave_inds] = np.array(gestalt_color_mapping['eave_end_point'])

    pcd.points = o3d.utility.Vector3dVector(triangulated_points)
    # print(colors)
    pcd.colors = o3d.utility.Vector3dVector(colors/255.0)
    
    return pcd

def get_open3d_lines(verts, edges):
    o3d_lineset = o3d.geometry.LineSet()
    o3d_lineset.points = o3d.utility.Vector3dVector(np.array(verts).reshape(-1,3))
    o3d_lineset.lines = o3d.utility.Vector2iVector(np.array(edges).reshape(-1,2))
    return o3d_lineset


def update_visualization_add_triangulated_point(vis, new_point, new_color):
    """
    Update the visualization by adding a new point
    :param geometries: list of geometries
    :param new_point: new point to add
    :param new_color: color of the new point
    :return: updated list of geometries
    """
    new_point = np.array(new_point).reshape(1,3)
    new_color = np.array(new_color).reshape(1,3)
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_point)
    new_pcd.colors = o3d.utility.Vector3dVector(new_color)

    vis.add_geometry(new_pcd)
    vis.update_renderer()

    return False


def visualize_3d_line_debug(sfm_points, gt_wf_vertices, gt_wf_edges, pred_wf_vertices, pred_wf_vertices_classes, points_scrutiny, horizontal_directions):

    # lines_scrutiny contains N points - the first point is connected to all the remaining points 
    line_3d = o3d.geometry.LineSet()
    line_3d.points = o3d.utility.Vector3dVector(points_scrutiny)
    edges = np.array([[0,i] for i in range(1, points_scrutiny.shape[0])])
    line_3d.lines = o3d.utility.Vector2iVector(edges)
    colors = np.array([[0,0,1]]*len(edges))
    line_3d.colors = o3d.utility.Vector3dVector(colors)

    o3d_major_directions = o3d.geometry.LineSet()
    origin = np.array([0, 0, 0])
    end_points = origin + 1000*np.array([horizontal_directions[0], horizontal_directions[1], [0, 0, 1]])
    all_points = np.vstack([origin, end_points])
    o3d_major_directions.points = o3d.utility.Vector3dVector(all_points)
    o3d_major_directions.lines = o3d.utility.Vector2iVector([[0, 1], [ 0, 2], [0, 3]])
    # color the major directions r g b
    o3d_major_directions.colors = o3d.utility.Vector3dVector(np.array([[255.0, 0, 0], [0, 255.0, 0], [0, 0, 255.0]])/255.0)

    o3d_gt_wf = o3d.geometry.LineSet()
    o3d_gt_wf.points = o3d.utility.Vector3dVector(np.array(gt_wf_vertices))
    o3d_gt_wf.lines = o3d.utility.Vector2iVector(np.array(gt_wf_edges))


    o3d_sfm_points = o3d.geometry.PointCloud()
    o3d_sfm_points.points = o3d.utility.Vector3dVector(sfm_points)
    o3d_sfm_points.paint_uniform_color([0.5, 0.5, 0.5])

    o3d_pred_points = get_triangulated_pts_o3d_pc(pred_wf_vertices, pred_wf_vertices_classes)

    o3d.visualization.draw_geometries([o3d_sfm_points, o3d_gt_wf, o3d_major_directions, o3d_pred_points, line_3d])

