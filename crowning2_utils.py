import os
import zipfile
import trimesh
import io
import shutil
import numpy as np
from trimesh import Trimesh
from vedo import Mesh
import tempfile
import pyvista as pv
import pyvista.plotting.colors as colors
from itertools import chain


color_values_list = list(colors.color_names.values())


def cross_section(mesh, plane_origin=[0,0,0], plane_normal=[0,0,0]):
    slice_ = mesh.section(plane_origin=plane_origin, 
                          plane_normal=plane_normal)
    to_2D = trimesh.geometry.align_vectors(plane_normal, [0,0,1]) 
    slice_2D, to_3D = slice_.to_planar(to_2D = to_2D)
    return slice_2D, to_3D

def get_designer_tooth2(udx_dirpath_):
    # non zip file
    data_ = None
    # try:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(udx_dirpath_, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    for file in os.listdir(temp_dir):
        if file.lower().endswith('zip'):
            fpath = os.path.join(temp_dir, file)
            with zipfile.ZipFile(fpath, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            os.remove(fpath)

    myfiles = list(chain.from_iterable([f for r,d,f in os.walk(temp_dir)]))

    for file1 in myfiles:
        if file1.lower().endswith('stl') and file1.split('.')[0][-1].isdigit():
            fpath2 = os.path.join(temp_dir, file1)
            data_ = trimesh.load_mesh(fpath2)
            break
    # except Exception as e:
    #     print(f'get_designer_tooth Exception: {e}')
        
    # finally:
    shutil.rmtree(temp_dir)
    return data_

def vedo_to_trimesh(vedo_mesh):
    vertices = vedo_mesh.points()
    faces = vedo_mesh.faces()
    trimesh_mesh = Trimesh(
        vertices=vertices,
        faces=faces,
        validate=True,
        process=False
    )
    return trimesh_mesh

def trimesh_to_vedo(trimesh_mesh):
    vertices = trimesh_mesh.vertices
    faces = trimesh_mesh.faces
    vedo_mesh = Mesh([vertices, faces])       
    return vedo_mesh

def pyvista_to_trimesh(pyvista_mesh):
    # Extract vertices and faces from the PyVista mesh
    vertices = pyvista_mesh.points
    faces = pyvista_mesh.faces.reshape(-1, 4)[:, 1:]
    
    # Create a trimesh using the extracted vertices and faces
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    return trimesh_mesh

def get_max_plane_idx(arr):
    absolute_differences = np.abs(arr[0] - arr[1])
    max_difference_index = np.argmax(absolute_differences)
    return max_difference_index

def no_intersect_ax(mesh_small_):
    ray_len = 10
    ray_endpoints = []
    ray_endpoints_neg = []
    start = mesh_small_.centroid
    for i in range(3):
        ray_endpoint = mesh_small_.centroid + np.array(
            [ray_len if j == i else 0 for j in range(3)])
        ray_endpoint2 = mesh_small_.centroid + np.array(
            [-ray_len if j == i else 0 for j in range(3)])
        ray_endpoints.append(ray_endpoint)
        ray_endpoints_neg.append(ray_endpoint2)

    for i,pt in enumerate(ray_endpoints):
        intercept, _ = pv.wrap(mesh_small_).ray_trace(start, pt)
        if not len(intercept.flatten()):
            if i == 0:
                return 'x'
            elif i == 1:
                return 'y'
            elif i == 2:
                return 'z'
                
    for i,pt2 in enumerate(ray_endpoints_neg):
        intercept, _ = pv.wrap(mesh_small_).ray_trace(start, pt2)
        if not len(intercept.flatten()):
            if i == 0:
                return 'x_neg'
            elif i == 1:
                return 'y_neg'
            elif i == 2:
                return 'z_neg'

def get_largest_axis_idx(bounds):
    diff = abs(bounds[0] - bounds[1])
    max_val = max(diff)
    return list(diff).index(max_val)

def get_nxt_largest_axis_idx(bounds):
    diff = abs(bounds[0] - bounds[1])
    return list(diff).index(sorted(diff)[-2])

def angle_between_vectors(vector_u, vector_v):
    cos_theta = np.dot(vector_u, vector_v) / (np.linalg.norm(vector_u) * np.linalg.norm(vector_v))
    angle = np.arccos(np.clip(cos_theta, -1, 1))
    return np.degrees(angle)

def calculate_distance(point1, point2):
    distance = np.sqrt(np.sum((point2 - point1)**2))
    return distance

def compute_angle(point1, point2):
    p1 = np.array(point1)
    p2 = np.array(point2)
    vector = p2 - p1
    norm_vector = vector / np.linalg.norm(vector)
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    angle_to_x_axis = np.arccos(np.dot(norm_vector, x_axis)) * 180 / np.pi
    angle_to_y_axis = np.arccos(np.dot(norm_vector, y_axis)) * 180 / np.pi
    angle_to_z_axis = np.arccos(np.dot(norm_vector, z_axis)) * 180 / np.pi
    
    angles_dict = {
        'x': angle_to_x_axis,
        'y': angle_to_y_axis,
        'z': angle_to_z_axis
    }
    
    return angles_dict