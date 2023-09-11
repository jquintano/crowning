def solidify_base(file):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(save_directory + "/" + file)
    
    #solidify mesh
    ms.meshing_close_holes(maxholesize = 10000, newfaceselected = False, selfintersection = False)
    ms.generate_splitting_by_connected_components(delete_source_mesh = True)
    
    target_prep_min = None
    target_prep_max = None
    num_vert_min = 1000000000
    num_vert_max = 0
    num_mesh = ms.number_meshes()
    
    #remove outliers
    for i in range(1, num_mesh+1):
        ms.set_current_mesh(i)
        #if the vertex number of current mesh is within the range, then it is an outlier
        if (ms.current_mesh().vertex_number() > 0 and ms.current_mesh().vertex_number() < 5000):
            ms.delete_current_mesh()
            continue
        if(ms.current_mesh().vertex_number() < num_vert_min):
            num_vert_min = ms.current_mesh().vertex_number()
            target_prep_min = i

        if (ms.current_mesh().vertex_number() > num_vert_max):
            num_vert_max = ms.current_mesh().vertex_number()
            target_prep_max = i
            
    for i in range(1, num_mesh+1):
        if(ms.mesh_id_exists(id = i)):
            if i!= target_prep_max and i!= target_prep_min :
                ms.set_current_mesh(i)
                ms.delete_current_mesh()
                
    #merge the base and prep
    ms.generate_boolean_union(first_mesh = target_prep_min, second_mesh = target_prep_max)
    ms.save_current_mesh(solidified_base + "/" + file)