import io
import trimesh
from matplotlib.colors import ListedColormap
import os
import tempfile
import zipfile
import numpy as np
import time
import shutil 
import pyvista as pv
import random
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
import trimesh
import itertools

udx_dirpath = r"C:\Users\ecejg\Downloads\ottawa_norris\udx_export\08142023"
udx_dirpath2 = r"C:\Users\ecejg\Downloads\ottawa_norris\udx_export\08272023"
tooth_dirpath = "./data/toothlib/mcaudie/post/"
tooth_dirpath_ottawa = "/Users/nimbyx/Projects/CROWNING/Ottawa Posterior"

def process_plot_case(udx_dirpath, tooth_dirpath):
    my_case = get_random_case(udx_dirpath)
    my_filename, my_mesh = load_prep_from_zip(my_case)
    my_mesh_split = get_two_largest_meshes_3m(my_mesh)
##    print(os.path.basename(my_case))
    p1 = pv.Plotter(shape=(1, 2))

    designed_tooth_mesh = get_designer_tooth(my_case)
    if designed_tooth_mesh:
        p1.subplot(0, 0)
        p1.add_mesh(pv.wrap(my_mesh))
        p1.add_mesh(designed_tooth_mesh, color='gold')
        p1.add_text('Designer', position='upper_right')
        p1.add_text(f"{my_filename}", position='lower_left')
        p1.camera_position = 'zx'
    else:
        p1.subplot(0, 0)
        p1.add_text('Not Available', position='lower_edge')
        p1.add_text('Designer', position='upper_right')

    my_case_details = get_toothnum(my_case)
##    print(my_case_details)
    mesh1, mesh2 = get_two_largest_meshes_3m(my_mesh)
    target_coords = np.array(mesh2.centroid)
    tooth_filepath = os.path.join(tooth_dirpath, f"{my_case_details[1]}.stl")

    if int(my_case_details[1]) in list(range(1,5,1)):
        scaler_val = 1.2
        tooth_mesh = pv.read(tooth_filepath
                            ).scale(np.full(3, scaler_val), inplace=True
                                   ).rotate_y(10, inplace=True
                                             ).rotate_z(-15, inplace=True
                                                       ).rotate_x(-20, inplace=True)                                                  
    
    elif int(my_case_details[1]) in list(range(13,17,1)):
        scaler_val = 1.1
        tooth_mesh = pv.read(tooth_filepath
                            ).scale(np.full(3, scaler_val), inplace=True
                                   ).rotate_y(-90, inplace=True
                                             ).rotate_z(-10, inplace=True
                                                       ).rotate_x(0, inplace=True)
    elif int(my_case_details[1]) in list(range(17,21,1)):
        scaler_val = 1.05
        tooth_mesh = pv.read(tooth_filepath
                            ).scale(np.full(3, scaler_val), inplace=True
                                   ).rotate_y(80, inplace=True
                                             ).rotate_z(5, inplace=True
                                                       ).rotate_x(-15, inplace=True)
    
    elif int(my_case_details[1]) in list(range(28,33,1)):
        scaler_val = 1.18
        tooth_mesh = pv.read(tooth_filepath
                            ).scale(np.full(3, scaler_val), inplace=True
                                   ).rotate_y(-105, inplace=True
                                             ).rotate_z(-25, inplace=True
                                                       ).rotate_x(-20, inplace=True)
    else:
        tooth_mesh = pv.read(tooth_filepath)

    tooth_coords = np.array(tooth_mesh.center)
    move_coords = target_coords - tooth_coords
    tooth_mesh.translate(move_coords, inplace=True)

    p1.subplot(0, 1)
    p1.add_mesh(pv.wrap(my_mesh))
    p1.add_text(f"{my_filename}", position='lower_left')
    p1.add_mesh(tooth_mesh, color='w')
    p1.add_text(f"tooth#{my_case_details[1]}", position='lower_right')
    p1.add_text('AI  ', position='upper_right')
    p1.camera_position = 'zx'

    p1.show()
    return my_filename


def plot_meshes_in_grid(udx_dirpath, tooth_dirpath, shape=(1, 2)):
    global rand_no_ls
    p = pv.Plotter(shape=shape)

    combinations = list(itertools.product(range(shape[0]), range(shape[1])))
    case_dir_ls = [os.path.join(udx_dirpath, f) for f in os.listdir(udx_dirpath)]
    rand_no_ls = []

    for c in combinations:
        rand_idx = random.randint(0, len(case_dir_ls) - 1)
        while rand_idx in rand_no_ls:
            rand_idx = random.randint(0, len(case_dir_ls) - 1)
        
        rand_no_ls.append(rand_idx)
        my_case = case_dir_ls[rand_idx]
        my_filename, my_mesh = load_prep_from_zip(my_case)

        my_case_details = get_toothnum(my_case)
        mesh1, mesh2 = get_two_largest_meshes_3m(my_mesh)
        target_coords = np.array(mesh2.centroid)
        tooth_filepath = os.path.join(tooth_dirpath, f"{my_case_details[1]}.stl")

        if int(my_case_details[1]) in list(range(1,5,1)):
            scaler_val = 1.2
            tooth_mesh = pv.read(tooth_filepath
                                ).scale(np.full(3, scaler_val), inplace=True
                                       ).rotate_y(10, inplace=True
                                                 ).rotate_z(-15, inplace=True
                                                           ).rotate_x(-20, inplace=True)                                                  
        elif int(my_case_details[1]) in list(range(13,17,1)):
            scaler_val = 1.1
            tooth_mesh = pv.read(tooth_filepath
                                ).scale(np.full(3, scaler_val), inplace=True
                                       ).rotate_y(-90, inplace=True
                                                 ).rotate_z(-10, inplace=True
                                                           ).rotate_x(0, inplace=True)
        elif int(my_case_details[1]) in list(range(17,21,1)):
            scaler_val = 1.05
            tooth_mesh = pv.read(tooth_filepath
                                ).scale(np.full(3, scaler_val), inplace=True
                                       ).rotate_y(90, inplace=True
                                                 ).rotate_z(5, inplace=True
                                                           ).rotate_x(-15, inplace=True)
        
        elif int(my_case_details[1]) in list(range(28,33,1)):
            scaler_val = 1.18
            tooth_mesh = pv.read(tooth_filepath
                                ).scale(np.full(3, scaler_val), inplace=True
                                       ).rotate_y(-105, inplace=True
                                                 ).rotate_z(-25, inplace=True
                                                           ).rotate_x(-20, inplace=True)

        tooth_coords = np.array(tooth_mesh.center)
        move_coords = target_coords - tooth_coords
        tooth_mesh.translate(move_coords, inplace=True)

        p.subplot(c[0], c[1])
        p.add_mesh(pv.wrap(my_mesh))
        p.add_text(f"{my_filename}", position='lower_left', font_size=10)
        p.add_mesh(tooth_mesh, color='w')
        p.add_text(f"tooth#{my_case_details[1]}", position='lower_right')
        p.add_text('AI', position='upper_right')
        p.camera_position = 'zx'
        # p.show_grid(show_xaxis=True, show_yaxis=True, show_zaxis=True)

    p.show()

def get_random_case(udx_dirpath_):
    return os.path.join(udx_dirpath_, os.listdir(udx_dirpath_)[random.randint(0, len(os.listdir(udx_dirpath_)))-1])

def get_designer_tooth(udx_dirpath_):
    data_ = None
    try:
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
##            if file1.lower().endswith('stl') and (file1.split('.')[0][-1] == '0':
            if file1.lower().endswith('stl') and file1.split('.')[0][-1].isdigit():
                fpath2 = os.path.join(temp_dir, file1)
                data_ = pv.read(fpath2)
                break
    except Exception as e:
        print(f'get_designer_tooth Exception: {e}')
        
    finally:
        shutil.rmtree(temp_dir)
        return data_

def get_toothnum(zip_file_path):
    data_ = None
    try:
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    
        for file in os.listdir(temp_dir):
            if file.lower().endswith('zip'):
                
                fpath = os.path.join(temp_dir, file)
                with zipfile.ZipFile(fpath, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                os.remove(fpath)
    
        # for root, dirs, files in os.walk(temp_dir):
        myfiles = list(chain.from_iterable([f for r,d,f in os.walk(temp_dir)]))

        for file1 in myfiles:
            if file1.lower().endswith('pts') and file1.split('.')[0][-2:].isdigit():
                tooth_no = int(file1.split('.')[0][-2:])
                file_name =  os.path.basename(zip_file_path)
                data_ = tuple([file_name, tooth_no])
                # data_ = {
                #     'FILENAME': file_name,
                #     'TOOTH_NUM': int(tooth_no)
                # }
                break
            elif file1.lower().endswith('dcm') and file1.split('.')[1][-2:].isdigit():
                tooth_no = file1.split('.')[1][-2:]
                file_name =  os.path.basename(zip_file_path)
                data_ = tuple([file_name, tooth_no])
                # data_ = {
                #     'FILENAME': file_name,
                #     'TOOTH_NUM': int(tooth_no)
                # }
                break
            elif file1.lower().endswith('dcm') and file1.split('.')[1][-1].isdigit():
                tooth_no = file1.split('.')[1][-1]
                file_name =  os.path.basename(zip_file_path)
                data_ = tuple([file_name, tooth_no])
                # data_ = {
                #     'FILENAME': file_name,
                #     'TOOTH_NUM': int(tooth_no)
                # }
                break
    except Exception as e:
        print(f'get_toothnum Exception: {e}')
        
    finally:
        shutil.rmtree(temp_dir)
        return data_

def gen_pie_chart(csv_file):
    df = pd.read_csv(csv_file)
    top5 = df.head(4)
    others = pd.DataFrame(
        {'Name': ['Others'], 'LabId': [0], 'TotalCount': [df['TotalCount'][5:].sum()], 
         'Percentage': [df['Percentage'][5:].sum()]})
    new_df = pd.concat([top5, others])
    cmap = plt.get_cmap("tab20c")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cmap(range(len(new_df)))
    wedges, texts, autotexts = ax.pie(
        new_df['TotalCount'],
        autopct='%1.1f%%',
        startangle=91,
        colors=colors,
        wedgeprops=dict(width=0.4),
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    center_circle = plt.Circle((0, 0), .67, fc='white')
    fig.gca().add_artist(center_circle)
    plt.subplots_adjust(top=1.1)
    plt.axis('equal')
    title = plt.title("2023 FCZ Posterior Single-Unit Cases", fontsize=25, fontweight='bold')
    legend = plt.legend(wedges, new_df['Name'], loc="center", bbox_to_anchor=(.28, 0, 0.5, 1))
    for text in legend.get_texts():
        text.set_fontsize(10)
        text.set_fontweight('bold')
    ax.text(-1.5, -1, f"Total Cases: {df['TotalCount'].sum()}", fontsize=10, color='gray')
    plt.show()

def gen_pie_chart2(csv_file):
    df = pd.read_csv(csv_file)
    significant_num = 4
    top5 = df.head(significant_num)
    others = pd.DataFrame(
        {'Name': ['Others'], 'LabId': [0], 'TotalCount': [df['TotalCount'][significant_num:].sum()], 
         'Percentage': [df['Percentage'][significant_num:].sum()]})
    new_df = pd.concat([top5, others])
    cmap = plt.get_cmap("tab20c")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = cmap(range(len(new_df)))
    wedges, texts, autotexts = ax.pie(
        new_df['TotalCount'],
        autopct='%1.1f%%',
        startangle=91,
        colors=colors,
        wedgeprops=dict(width=0.4),
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    center_circle = plt.Circle((0, 0), .67, fc='white')
    fig.gca().add_artist(center_circle)
    plt.subplots_adjust(top=1.1)
    plt.axis('equal')
    title = plt.title("2023 FCZ Posterior Single-Unit Cases", fontsize=25, fontweight='bold')
    legend = plt.legend(wedges, new_df['Name'], loc="center", bbox_to_anchor=(.28, 0, 0.5, 1))
    for text in legend.get_texts():
        text.set_fontsize(10)
        text.set_fontweight('bold')
    ax.text(-1.5, -1, f"Total Cases: {df['TotalCount'].sum()}", fontsize=10, color='gray')
    plt.show()

def get_tooth_details(udx_path): 
    data = []
    for udx in os.listdir(udx_path):
        temp_dir = tempfile.mkdtemp()
        dl_path = os.path.join(udx_path, udx)
        if os.path.isfile(dl_path):
            with zipfile.ZipFile(dl_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            if os.listdir(temp_dir):
                for file in os.listdir(temp_dir):
                    try:
                        if file.endswith('pts') and file.split('.')[0][-2:].isdigit():
                            tooth_no = file.split('.')[0][-2:]
                            file_name = file.split('_00_')[0]
                            
                            data_ = {
                                'FILENAME': f'{file_name}.zip',
                                'TOOTH_NUM': int(tooth_no)
                            }
                            
                            data.append(data_)
                    except Exception as e:
                        print(f'{file} Exception: {e}')
                        pass
        shutil.rmtree(temp_dir)
        
    df = pd.DataFrame(data)
    return df


def load_prep_from_zip(file_path):
    stl_file_name = "PreparationScan.stl"
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        with zip_ref.open(stl_file_name) as file:
            bytes_buffer = io.BytesIO(file.read())
            mesh_3m = trimesh.load_mesh(bytes_buffer, file_type='stl')
            return os.path.basename(file_path), mesh_3m

def load_anta_from_zip(file_path):
    stl_file_name = "AntagonistScan.stl"
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        with zip_ref.open(stl_file_name) as file:
            bytes_buffer = io.BytesIO(file.read())
            mesh_3m = trimesh.load_mesh(bytes_buffer, file_type='stl')
            return os.path.basename(file_path), mesh_3m

def trimesh_to_vedo(trimesh_mesh):
    vertices = trimesh_mesh.vertices
    faces = trimesh_mesh.faces
    vedo_mesh = Mesh([vertices, faces])       
    return vedo_mesh

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

def closest_elements(numbers):
    if len(numbers) < 2:
        raise ValueError("List must contain at least two numbers")

    numbers.sort()  # Sort the list in ascending order
    min_difference = float('inf')
    closest_pair = None

    for i in range(len(numbers) - 1):
        difference = abs(numbers[i + 1] - numbers[i])
        if difference < min_difference:
            min_difference = difference
            closest_pair = (numbers[i], numbers[i + 1])

    return closest_pair

def closest_element_indices(numbers):
    if len(numbers) < 2:
        raise ValueError("List must contain at least two numbers")

    sorted_indices = sorted(range(len(numbers)), key=lambda i: numbers[i])
    min_difference = float('inf')
    closest_indices = None

    for i in range(len(sorted_indices) - 1):
        index1 = sorted_indices[i]
        index2 = sorted_indices[i + 1]
        difference = abs(numbers[index2] - numbers[index1])

        if difference < min_difference:
            min_difference = difference
            closest_indices = (index1, index2)
    return closest_indices

def compute_range(tup):
    if len(tup) != 6:
        return
    return abs(tup[0]-tup[1]), abs(tup[2]-tup[3]), abs(tup[4]-tup[5])

# def get_two_largest_meshes_3m(mesh):
#     components = mesh.split(only_watertight=False)
#     components_sorted = sorted(components, key=lambda x: len(x.faces), reverse=True)
#     if len(components_sorted) < 2:
#         return components_sorted
#     return components_sorted[:2]

def get_two_largest_meshes_3m(my_trimesh):
    my_meshes = my_trimesh.split(only_watertight=False)
    mesh_vol = []   
    for i, mesh in enumerate(my_meshes):
        bounds = mesh.bounds
        box_dimensions = bounds[1] - bounds[0]
        volume = np.prod(box_dimensions)
        mesh_vol.append((i, volume))
    sorted_mesh_vol = sorted(mesh_vol, key=lambda x: x[1], reverse=True)
    first_largest_index = sorted_mesh_vol[0][0]
    second_largest_index = sorted_mesh_vol[1][0]
    return my_meshes[first_largest_index], my_meshes[second_largest_index]
