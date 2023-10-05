from demo2_utils import *
from crowning2_utils import *

def gen_crown1(zip_order_path, toothlib_scale=1):
    print(f'Run: gen_crown1')
    my_crown = None
    mesh_prep_fname, mesh_prep_3m = load_prep_from_zip(zip_order_path)
    mesh_big, mesh_small = get_two_largest_meshes_3m(mesh_prep_3m)
    my_case_details = get_toothnum(zip_order_path)
    tooth_filepath = os.path.join(tooth_dirpath_ottawa, f"{my_case_details[1]}.stl")
    tooth_mesh = pv.read(tooth_filepath)
    largest_plane = get_largest_axis_idx(mesh_prep_3m.bounds)
    # print(f'largest plane: {largest_plane}')
    if largest_plane in [0,2]:
        if largest_plane == 2:
            tooth_mesh1 = get_horizontal_position(mesh_prep_3m.bounds, mesh_small.centroid, tooth_mesh)
        elif largest_plane == 0:
            tooth_mesh1 = get_vertical_position(mesh_prep_3m.bounds, mesh_small.centroid, tooth_mesh)
        tooth_mesh2 = tooth_mesh1.scale(np.full(3, 1.00))
        tooth_mesh3 = tooth_mesh2.translate(gen_move_coords(mesh_small.centroid, tooth_mesh2.center))
        my_crown = tooth_mesh3.boolean_difference(pv.wrap(mesh_small)).fill_holes(5)
    else:
        print('y-plane largest')
    return my_crown

def plot_crown(gen_crown, zip_order_path):
    mesh_prep_fname, mesh_prep_3m = load_prep_from_zip(zip_order_path)
    mesh_big, mesh_small = get_two_largest_meshes_3m(mesh_prep_3m)
    cam_pos = 'xy'
    my_shape = (1,2)
    p1_ = pv.Plotter(shape=my_shape)
    p1_.subplot(0,0) #1st plot
    p1_.add_mesh(mesh_small)
    p1_.add_mesh(mesh_big)
    p1_.add_mesh(my_crown, color='gold', opacity=.7)
    p1_.add_text(f'{my_case_details[0]}', position='lower_right', font_size=10)
    p1_.camera_position=cam_pos
    p1_.show_grid()
    p1_.subplot(0,1) #2nd plot
    p1_.add_mesh(my_crown, color='gold')
    p1_.add_text(f'{my_case_details[1]}', position='upper_right')
    p1_.add_text('antagonist', position='lower_right', font_size=10)
    p1_.camera_position=cam_pos
    p1_.show()