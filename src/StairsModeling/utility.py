import numpy as np


def add_to_multipointcloud_multicolor(p, clouds, colors):
    p_multicolor = p
    for cloud, color in zip(clouds, colors):
        p_multicolor = add_to_pointcloud_color(p_multicolor, cloud, color)
    return p_multicolor

def generate_plane_frame_batch_multicolor(plane_points, direction_vector, plane_normal, size, colors):
    plane_frames = []
    for plane_point in plane_points:
        plane_frame = generate_plane_frame(plane_point, direction_vector,
                                                    plane_normal, size)      
        plane_frames.append(plane_frame)
    plane_frames_multicolor = add_to_multipointcloud_multicolor(None, plane_frames, colors)
    return plane_frames_multicolor

def generate_plane_frame_batch(plane_points, direction_vector, plane_normal, size):
    plane_frames = None
    for plane_point in plane_points:
        plane_frame = generate_plane_frame(plane_point, direction_vector,
                                                    plane_normal, size)
        plane_frames = add_to_pointcloud(plane_frames, plane_frame)
    return plane_frames

def generate_plane_frame(plane_point, direction_vector, plane_normal, size):
    w,h = size
    right_center = plane_point + direction_vector*w/2
    left_center = plane_point - direction_vector*w/2
    line_normal = np.cross(direction_vector, plane_normal)
    top_center = plane_point + line_normal*h/2
    bottom_center = plane_point - line_normal*h/2
    left_border_points = left_center + np.outer(np.arange(-h/2.,h/2.,0.01), line_normal)
    right_border_points = right_center + np.outer(np.arange(-h/2.,h/2.,0.01), line_normal)
    top_border_points = top_center + np.outer(np.arange(-w/2.,w/2.,0.01), direction_vector)
    bottom_border_points = bottom_center + np.outer(np.arange(-w/2.,w/2.,0.01), direction_vector)
    all_points = np.vstack((left_border_points,right_border_points,top_border_points,bottom_border_points))
    return all_points

def add_to_pointcloud_color(p, p1, color):
    p1_color = paint_pointcloud(p1, color)
    p = add_to_pointcloud(p, p1_color)
    return p

def add_to_pointcloud(p, p1):
    if p is None:
        p = p1
    else:
        p = np.vstack((p,p1))
    return p

def paint_pointcloud(p, color):
    p_color = np.hstack((p, np.ones((p.shape[0],1))*color_to_float(color)))
    return p_color

def write_XYZRGB(points, filename):
    header = '# .PCD v0.7 - Point Cloud Data file format\n\
VERSION 0.7\n\
FIELDS x y z rgb\n\
SIZE 4 4 4 4\n\
TYPE F F F F\n\
COUNT 1 1 1 1\n\
WIDTH %d\n\
HEIGHT 1\n\
VIEWPOINT 0 0 0 1 0 0 0\n\
POINTS %d\n\
DATA ascii\n' % (points.shape[0], points.shape[0])
    f = open(filename, 'w')
    f.write(header)
    for p in points: 
        f.write(' '.join([str(v) for v in p]) + '\n')
    f.close()


def color_to_float(color):
    import struct
    if color.size == 1:
        color = [color] * 3
    rgb = (color[2] << 16 | color[1] << 8 | color[0]);
    rgb_hex = hex(rgb)[2:-1]
    s = '0' * (8 - len(rgb_hex)) + rgb_hex.capitalize()
#            print color, rgb, hex(rgb)
    rgb_float = struct.unpack('!f', s.decode('hex'))[0]
#            print rgb_float
    return rgb_float
