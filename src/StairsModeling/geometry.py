import numpy as np
import numpy.linalg as linalg

def sample_vector_normal_to_vector(axis, size):
    '''
    axis = [a,b,c], unit directional vector
    initial guess v0 is the closest to (0,0,1) 
    '''
    proj_vector = project_to_plane(np.array([0,0,1]), axis, np.array([0,0,0]))
    v0 = proj_vector/linalg.norm(proj_vector)
#    v0 = np.array([axis[1],-axis[0],0], dtype=np.float)
#    thetas = np.random.random(size)*np.pi
    thetas = np.random.random(size)*20*np.pi/180
    vrots = [rodrigues_rotation(v0, axis, sample_theta) for sample_theta in thetas]
    vrots = np.array(vrots, dtype=np.float)
    return vrots

def distance_origin_to_plane_batch(plane_normal, plane_points):
    '''
    plane_points: n*3
    '''
    d = -np.dot(plane_points, plane_normal)
    return abs(d)

def distance_origin_to_plane(plane_normal, plane_point):
    d = -np.dot(plane_normal, plane_point)
    return abs(d)

def is_on_same_side_as(target_point, line_direction, line_point, plane_normal, ref_point):
    '''
    Test whether an in-plane point is on the same side of a planar line as the in-plane ref_point 
    '''
    line_normal_on_plane = np.cross(plane_normal, line_direction)
    target_proj = project_to_plane(target_point, plane_normal, line_point)
    sign_target = np.dot(line_normal_on_plane, target_proj-line_point) > 0
    sign_ref = np.dot(line_normal_on_plane, ref_point-line_point) > 0
    return sign_target == sign_ref

def adjust_normal_direction(plane_normal, plane_point):
    '''
    Make sure all normal vectors are pointing away from the origin
    '''
    if np.dot(plane_point, plane_normal) > 0:
        return plane_normal
    else:
        return -plane_normal

def project_to_plane(target_point, plane_normal, plane_point):
    proj_len = np.dot(plane_point - target_point, plane_normal)
    proj = target_point - proj_len*plane_normal
    return proj

def project_to_plane_batch(target_points, plane_normal, plane_point):
    '''
    target_points: n by 3
    '''
    proj_lens = np.dot(plane_point - target_points, plane_normal)
    projs = target_points - np.outer(proj_lens, plane_normal)
    return projs

def rodrigues_rotation(v, axis, theta):
    vrot = v*np.cos(theta) + np.cross(axis, v)*np.sin(theta) + np.dot(axis, np.dot(axis,v)*(1-np.cos(theta)))
    return vrot

def plane_to_hessian_normal_form(plane_normal, plane_points):
    '''
    plane_points: n by 3
    '''
    d = -np.dot(plane_normal, plane_points)
    return np.append(plane_normal, d)

def point_to_plane_distance(target_points, plane_normal, plane_point):
    '''
    target_points: n by 3
    '''
    d = -np.dot(plane_point, plane_normal)
    dists = abs(np.dot(target_points, plane_normal)+d)/linalg.norm(plane_normal)
    return dists

def fit_plane_indices(p_arr, indices):
    selected_points = p_arr[indices,:]
    xs = selected_points[:,0]
    ys = selected_points[:,1]
    zs = selected_points[:,2]
    A = np.column_stack((xs, ys, np.ones(xs.size)))
    plane_params, resid,rank,sigma = np.linalg.lstsq(A,zs)
    normal = np.array([plane_params[0],plane_params[1],-1])
    normal = normal/linalg.norm(normal)
    return plane_params, normal

def distance_to_plane(p_arr, ind, plane_params):
    x, y, z = p_arr[ind]
    a,b,d = plane_params
    distance = abs(a*x + b*y + z + d)/np.sqrt(a**2+b**2+1+d**2)
    return distance 