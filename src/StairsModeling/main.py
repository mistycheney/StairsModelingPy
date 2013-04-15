'''
Created on Apr 5, 2013

@author: yuncong
'''

import os, sys
import numpy as np
from collections import OrderedDict
import random 
import numpy.linalg as linalg
from StairsModeling import Edge, StereoMatch, utility, geometry
import itertools
import pcl
import time

IMGPATH='/Users/yuncong/Documents/StairsModelingPy/staircase_new/'
PROJPATH='/Users/yuncong/Documents/StairsModelingPy/'

import cv2
img_id = 18
top = cv2.imread(IMGPATH+"top%d.jpg"%img_id, 0)
#top = cv2.resize(top_o, (top_o.shape[1]/2, top_o.shape[0]/2))
bottom = cv2.imread(IMGPATH+"bottom%d.jpg"%img_id, 0)
#bottom = cv2.resize(bottom_o, (bottom_o.shape[1]/2, bottom_o.shape[0]/2))

if __name__ == '__main__':
    
    os.chdir(PROJPATH)
    
    sgbmParams = OrderedDict([('SADWindowSize',(5,51)),
                              ('numberOfDisparitiesMultiplier',(11,1000)),
                              ('preFilterCap',(100,1000)),
                              ('minDisparity',(0,1000)),
                              ('uniquenessRatio',(3,20)),
                              ('speckleWindowSize',(0,1000)),
                              ('P1',(300,100000)),  #300 for raw
                              ('P2',(90000,100000)),  #90000 for raw
                              ('speckleRange',(1,10))])
    sgbm = StereoMatch.SGBMTuner(sgbmParams, 'SGBMTuner', top, bottom)
    
    xyz_valid = np.array([i for i in itertools.chain(*sgbm.xyz) if i[2] != 10000.]).astype(np.float32)
    print xyz_valid.shape
    
#    xyzrgb_valid = np.array([i for i in itertools.chain(*sgbm.xyzrgb) if i[2] != 10000.]).astype(np.float32)
#    print xyzrgb_valid.shape

#    xyzrgb = sgbm.xyzrgb
#    print xyz.size / 3
#    invalid = (sgbm.xyzrgb == -np.inf).any(axis=2)
#    xyzrgb_valid = sgbm.xyzrgb[np.nonzero(1-invalid)]
#    print sgbm.xyzrgb.shape
#    print xyzrgb_valid.shape
#    write_XYZRGB(xyzrgb_valid, 'xyzrgb_valid.pcd')
#    sys.exit()
    
    xyz = sgbm.xyz
    
    edgeParams = OrderedDict([('thresh1',(73,2000)),
                              ('thresh2',(137,2000)),
                            ('apertureSize',(3,21)),
                              ('hough_thresh',(20,500)),
                              ('minLineLength',(10,500)),
                              ('maxLineGap',(100,500)),  #38
                              ('rho',(1,50)),
                              ('theta',(1,10))])
    edge = Edge.EdgeTuner(edgeParams, 'EdgeTuner', sgbm.top_r)
    
    line_points = None
    disp_color = cv2.cvtColor(sgbm.disp8, cv2.cv.CV_GRAY2BGR)
#    step = 0.01
    merged_line_3d = []
    
    begin = time.time()
    for i, (rho,theta,x1,y1,x2,y2) in enumerate(edge.merged_line):
        x1p, y1p, x2p, y2p = int(x1), int(y1), int(x2), int(y2)
        p1 = np.array([x1p, y1p])
        p2 = np.array([x2p, y2p]) 
        cv2.line(disp_color, (x1p, y1p), (x2p, y2p), (0,0,255))
#        e1_3d = np.array(xyz[y1p, x1p])
#        e2_3d = np.array(xyz[y2p, x2p])
#        
#        print x1p, y1p, x2p, y2p
#        print 'e1', e1_3d, 'e2', e2_3d
        
        steps = np.arange(0,1,0.01)
        interm_pixels = (np.outer(steps, p2) + np.outer(1-steps, p1)).astype(np.int)
        interm_points_3d = xyz[interm_pixels[:,1],interm_pixels[:,0]]
        interm_points_3d_valid = np.array([p for p in interm_points_3d if -np.inf not in p])
        print 'total points', interm_points_3d.shape[0], 'valid points', interm_points_3d_valid.shape[0]
#        print interm_points_3d_valid
        # interm_points_3d_valid: n*3
        
        mean = np.mean(interm_points_3d_valid, axis=0)
        std =  np.std(interm_points_3d_valid, axis=0)
        interm_points_3d_valid_normalized = (interm_points_3d_valid - mean) / std
    
        U,s,Vt = linalg.svd(interm_points_3d_valid_normalized)
        V = Vt.T
        ind = np.argsort(s)[::-1]
        U = U[:,ind]
        s = s[ind]
        V = V[:,ind]
        S = np.diag(s)
        Mhat = np.dot(U[:,:1],np.dot(S[:1,:1],V[:,:1].T)) * std + mean
        projected_points = Mhat[Mhat[:,2].argsort()]
#        print projected_points
        center_point = (projected_points[0] + projected_points[-1])/2
        diff = projected_points[-1] - projected_points[0]
#        print projected_points[-1], projected_points[0], diff, linalg.norm(diff)
        direction_vector = diff / linalg.norm(diff)
        print 'center point', center_point, 'direction_vector', i, direction_vector
        
#        if np.dot(direction_vector, np.array([1,0,0])) > np.cos(10*np.pi/180):
        if np.dot(direction_vector, np.array([1,0,0])) > 0.95:
            print 'added'
            merged_line_3d.append(np.hstack((center_point, direction_vector)))

        if line_points is None:
            line_points = Mhat
        else:
#            line_points = np.vstack((line_points, Mhat,interm_points_3d_valid))
            line_points = np.vstack((line_points, Mhat))
    
    print 'PCA edge line time', time.time() - begin
    
    cv2.imshow("disp_color", disp_color)
    cv2.waitKey()
    
    #    utility.write_XYZRGB(line_points_color, 'edges.pcd')
     
     
#    edge_p = pcl.PointCloud()
#    edge_p.from_array(np.array(line_points))
#    os.chdir(PROJPATH)
#    edge_p.to_file("edges.pcd", ascii=True)
    
#    xyz_valid = np.array([i for i in itertools.chain(*xyz) if i[2] != 10000.]).astype(np.float32)
#    edge_array = np.vstack((np.array(line_points), xyz_valid))    
#    xyzrgb_valid = [color_to_float(c) for c in top]
    
#    print len(xyz_valid), len(edge_array)
#    edge_p = pcl.PointCloud()
#    edge_array = np.array(line_points)
#    edge_p.from_array(line_points)
#    edge_p.to_file("edges.pcd", ascii=True)

    p = pcl.PointCloud()
    p.from_array(xyz_valid)
    vox = p.make_voxel_grid_filter()
    vox.set_leaf_size(0.01,0.01,0.01)
    pv = vox.filter()
    xyz_downsampled = pv.to_array()
    point_number = xyz_downsampled.shape[0]
    print 'after voxel grid filter', point_number
    
#    normals = geometry.compute_cloud_normals(xyz_downsampled, 10)
#    normals_cloud = utility.draw_normals(xyz_downsampled, normals)
#    xyz_cloud = utility.paint_pointcloud(xyz_downsampled, np.array([0,0,255]))
#    xyz_normals_cloud = utility.add_to_pointcloud_color(xyz_cloud, normals_cloud, np.array([255,255,255]))
#    utility.write_XYZRGB(xyz_normals_cloud, 'xyz_normals_cloud.pcd')
#    sys.exit()
    
    merged_line_3d = np.array(merged_line_3d)
    plane_number = merged_line_3d.shape[0]
    direction_vector = np.mean(merged_line_3d[:,3:], axis=0)
    direction_vector = direction_vector/linalg.norm(direction_vector)
    print 'direction_vector', direction_vector
    edge_points = merged_line_3d[:,:3]
    sample_rise_normal = geometry.sample_vector_normal_to_vector(direction_vector, 1)
#    rise_colors = np.random.randint(0,255,(plane_number,3))
#    tread_colors = np.random.randint(0,255,(plane_number,3))
    plane_colors =  np.random.randint(0,255,(2*plane_number,3))
    plane_colors[0] = np.array([255,255,255])
    plane_colors[1] = np.array([200,200,200])

    for sample_rise_ind, rise_normal in enumerate(sample_rise_normal):
                
#        plane_frames = utility.generate_plane_frame_batch(merged_line_3d[:,:3], direction_vector, rise_normal,(4,2))
#        edges_plane_points = utility.add_to_pointcloud_color(line_points_color, plane_frames, np.array([255,0,0]))
#        utility.write_XYZRGB(all_points, 'edges_plane.pcd')
#        sys.exit()
        
        rise_normal = geometry.adjust_normal_direction(rise_normal, edge_points[0])
        tread_normal = np.cross(direction_vector, rise_normal)
        tread_normal = tread_normal/linalg.norm(tread_normal)
        tread_normal = geometry.adjust_normal_direction(tread_normal, edge_points[0])
        print 'rise_normal', rise_normal
        print 'tread_normal', tread_normal
        
#        rise_cloud = utility.generate_plane_frame_batch_multicolor(merged_line_3d[:,:3], 
#                                          direction_vector, rise_normal, (4,1), rise_colors)
#        tread_cloud = utility.generate_plane_frame_batch_multicolor(merged_line_3d[:,:3], 
#                                          direction_vector, tread_normal, (4,1), tread_colors)
#        plane_cloud = utility.add_to_pointcloud(rise_cloud, tread_cloud)
        stairs_plane_cloud = utility.generate_stairs_plane_frame_batch_multicolor(edge_points, direction_vector,
                                                                       rise_normal, tread_normal, plane_colors)
        edges_plane_cloud = utility.add_to_pointcloud_color(stairs_plane_cloud, line_points, np.array([255,0,0]))
        axis_cloud = utility.draw_axis()
        edges_plane_cloud = utility.add_to_pointcloud(edges_plane_cloud, axis_cloud)
        utility.write_XYZRGB(edges_plane_cloud, 'edges_plane.pcd')
        
#        sys.exit()
        
        rise_dist = np.zeros((plane_number-1,))
        tread_dist = np.zeros((plane_number-1,))
        for i in range(plane_number-1):
            rise_dist[i] = geometry.point_to_plane_distance([merged_line_3d[i,:3]], 
                                            rise_normal, merged_line_3d[i+1,:3])[0]
            tread_dist[i] = geometry.point_to_plane_distance([merged_line_3d[i,:3]], 
                                            tread_normal, merged_line_3d[i+1,:3])[0]
        
        rise_dist_sorted = rise_dist[rise_dist.argsort()]
        rise_multiples_ratio = rise_dist_sorted/rise_dist_sorted[0]
        rise_multiples = rise_multiples_ratio.astype(np.int)
        rise_dist_each = rise_dist_sorted/rise_multiples
        rise_dist_mean = rise_dist_each.mean()
        rise_dist_range = rise_dist_mean + rise_dist_each.std()*np.array([-1,1])
        
        tread_dist_sorted = tread_dist[tread_dist.argsort()]
        tread_multiples_ratio = tread_dist_sorted/tread_dist_sorted[0]
        tread_multiples = tread_multiples_ratio.astype(np.int)
        tread_dist_each = tread_dist_sorted/tread_multiples
        tread_dist_mean = tread_dist_each.mean()
        tread_dist_range = tread_dist_mean + tread_dist_each.std()*np.array([-1,1])
        
        print 'rise_dist', rise_dist
        print 'rise_multiples_ratio', rise_multiples_ratio
        print 'rise_dist_range', rise_dist_range
        print 'tread_dist', tread_dist
        print 'tread_multiples_ratio', tread_multiples_ratio
        print 'tread_dist_range', tread_dist_range
        
        rise_offset = geometry.signed_distance_origin_to_plane_batch(rise_normal, edge_points)
        tread_offset = geometry.signed_distance_origin_to_plane_batch(tread_normal, edge_points)
        print 'rise_offset', rise_offset
        print 'tread_offset', tread_offset
        
        sample_indices = np.random.randint(0, point_number, point_number/3)
#        sample_indices = range(point_number)
        sample_number = len(sample_indices)
        sample_points = xyz_downsampled[sample_indices]
#        for p in xyz_downsampled:
        print 'sample_points', sample_points

#        edges_plane_sample_cloud = utility.add_to_pointcloud_color(edges_plane_cloud, 
#                                                        sample_points, np.array([0,255,0]))
#        utility.write_XYZRGB(edges_plane_sample_cloud, 'edges_plane_sample.pcd')
#        sys.exit()
        
        p_rise_offset = geometry.signed_distance_origin_to_plane_batch(rise_normal, sample_points)
#        sample_rise_cloud = utility.generate_plane_frame_batch(sample_points, direction_vector, rise_normal, (4,1))
#        edges_plane_cloud = utility.add_to_pointcloud_color(edges_plane_cloud, sample_rise_cloud, np.array([0,0,255]))
        
        print 'p_rise_offset', p_rise_offset
        if rise_offset[0] > rise_offset[-1]:
            farther_than_rise = np.greater.outer(p_rise_offset, np.hstack((9999, rise_offset, -9999)))
        else:
            farther_than_rise = np.greater.outer(p_rise_offset, np.hstack((-9999, rise_offset, 9999)))
        print farther_than_rise
        row_id, front_rise = np.nonzero(farther_than_rise[:,1:] != farther_than_rise[:,:-1])    
        front_rise = front_rise-1
        
        
        
        p_tread_offset = geometry.signed_distance_origin_to_plane_batch(tread_normal, sample_points)
#        sample_tread_cloud = utility.generate_plane_frame_batch(sample_points, direction_vector, tread_normal, (4,1))
#        edges_plane_cloud = utility.add_to_pointcloud_color(edges_plane_cloud, sample_tread_cloud, np.array([0,0,255])) 
        
        print 'p_tread_offset', p_tread_offset
        if tread_offset[0] > tread_offset[-1]:
            farther_than_tread = np.greater.outer(p_tread_offset, np.hstack((9999, tread_offset, -9999)))
        else:
            farther_than_tread = np.greater.outer(p_tread_offset, np.hstack((-9999, tread_offset, 9999)))
        print farther_than_rise
        row_id, front_tread = np.nonzero(farther_than_tread[:,1:] != farther_than_tread[:,:-1])
        front_tread = front_tread-1
        
        print 'front_rise', front_rise, front_rise.size
        print 'front_tread', front_tread, front_tread.size
        
        planes_to_compare = np.nan*np.ones((sample_number,2))
        is_outside = front_tread == front_rise
        planes_to_compare[is_outside] = np.column_stack((front_tread[is_outside], front_rise[is_outside] + 1))
        is_inside = front_tread == front_rise-1
        planes_to_compare[is_inside] = np.column_stack((front_tread[is_inside]+1, front_rise[is_inside]))
        
#        print 'planes_to_compare', planes_to_compare
        in_roi = is_outside + is_inside
        sample_points_roi = sample_points[in_roi]
        planes_to_compare_roi = planes_to_compare[in_roi]
        roi_number = sample_points_roi.shape[0]  
        print 'planes_to_compare_roi', planes_to_compare_roi
        
#        edges_plane_sample_roi_cloud = utility.add_to_pointcloud_color(edges_plane_cloud, 
#                                                        sample_points_roi, np.array([0,255,0]))
#        utility.write_XYZRGB(edges_plane_sample_roi_cloud, 'edges_plane_sample_roi.pcd')

#        sys.exit()
        
        dist_to_plane = np.inf*np.ones((roi_number,2))    
        for edge_ind, edge_point in enumerate(merged_line_3d[:,:3]):
            print 'edge',edge_ind, edge_point
            is_comparing_this_plane = planes_to_compare_roi == edge_ind
            print is_comparing_this_plane 
            dist_to_plane[is_comparing_this_plane[:,1], 1] =\
             geometry.project_to_plane_only_distance_batch(sample_points_roi[is_comparing_this_plane[:,1]],
                                                           rise_normal, edge_point)
            dist_to_plane[is_comparing_this_plane[:,0], 0] =\
             geometry.project_to_plane_only_distance_batch(sample_points_roi[is_comparing_this_plane[:,0]],
                                              tread_normal, edge_point)
        print 'dist_to_plane', dist_to_plane
        attach_to_rise_not_tread = dist_to_plane[:,0] > dist_to_plane[:,1]
        attach_plane_dist = np.where(attach_to_rise_not_tread, dist_to_plane[:,1], dist_to_plane[:,0])
        attach_plane = np.where(attach_to_rise_not_tread, planes_to_compare_roi[:,1], planes_to_compare_roi[:,0]) 
        
#        def find_closest_plane_batch(rise_not_tread):
#            
#            if rise_not_tread:
#                print 'rise'
#                plane_normal = rise_normal
#                front_plane = front_rise
#            else:
#                print 'tread'
#                plane_normal = tread_normal
#                front_plane = front_tread
#                
#            sample_dist_to_front = 9999*np.ones((sample_number,))
#            sample_projs_to_front = np.zeros((sample_number,3))
##            sample_on_correct_side_front = np.zeros((sample_number,))
#            sample_dist_to_back = 9999*np.ones((sample_number,))
#            sample_projs_to_back = np.zeros((sample_number,3))
##            sample_on_correct_side_back = np.zeros((sample_number,))
#            for plane_ind, edge_point in enumerate(merged_line_3d[:,:3]):
#                print 'plane_ind', plane_ind
#                sample_indices_with_front = np.nonzero(plane_ind == front_plane)[0]
#                sample_indices_with_back = np.nonzero(plane_ind-1 == front_plane)[0]
#                print 'sample_indices_with_front', sample_indices_with_front
#                print 'sample_indices_with_back', sample_indices_with_back
#                
#                l1 = sample_indices_with_front.size
#                sample_indices_all_compute = np.hstack((sample_indices_with_front,
#                                                        sample_indices_with_back))
#                sample_projs, sample_dist =\
#                 geometry.project_to_plane_with_distance_batch(sample_points[sample_indices_all_compute],
#                                              plane_normal, edge_point)
#                
#                sample_projs_to_front[sample_indices_with_front] = sample_projs[:l1]
#                sample_dist_to_front[sample_indices_with_front] = sample_dist[:l1]
#                sample_projs_to_back[sample_indices_with_back] = sample_projs[l1:]
#                sample_dist_to_back[sample_indices_with_back] = sample_dist[l1:]
            
#                if rise_not_tread:
#                    ref_point = np.array([0,-999,0])
#                else:
#                    ref_point = np.array([0,0,-999])
#                sample_on_correct_side =\
#                 geometry.is_on_same_side_as_with_projs_batch(sample_points[sample_indices_all_compute],
#                 direction_vector, edge_point, plane_normal, ref_point,
#                   sample_projs)
#                sample_on_correct_side_front[sample_indices_with_front] = sample_on_correct_side[:l1]
#                sample_on_correct_side_back[sample_indices_with_back] = sample_on_correct_side[l1:]           
            
#            print 'sample_dist_to_front', sample_dist_to_front
##            print 'sample_on_correct_side_front', sample_on_correct_side_front
#            print 'sample_dist_to_back', sample_dist_to_back
#            print 'sample_on_correct_side_back', sample_on_correct_side_back
            
#            if rise_not_tread:
##                closer_to_front_not_back = True
#                closest_plane = front_plane
#                closest_dist = sample_dist_to_front
##            print 'closer_to_front_not_back', closer_to_front_not_back
#            else:
##                closer_to_front_not_back = False
#                closest_plane = front_plane + 1
#                closest_dist = sample_dist_to_back
##            closest_plane = np.where(closer_to_front_not_back, front_plane, front_plane+1)
##            closest_dist = np.where(closer_to_front_not_back, sample_dist_to_front,
##                                   sample_dist_to_back)
#            return closest_plane, closest_dist
#            on_correct_side = np.where(closer_to_front_not_back, sample_on_correct_side_front, 
#                                       sample_on_correct_side_back)
#            return closest_plane, closest_dist, on_correct_side
            
#        closest_rise, closest_dist_rise, on_correct_side_rise = find_closest_plane_batch(True)
#        closest_rise, closest_dist_rise = find_closest_plane_batch(True)
#        print 'closest_rise', closest_rise
#        print 'closest_dist_rise', closest_dist_rise
##        print 'on_correct_side_rise', on_correct_side_rise
##        closest_tread, closest_dist_tread, on_correct_side_tread = find_closest_plane_batch(False)
#        closest_tread, closest_dist_tread = find_closest_plane_batch(False)
#        print 'closest_tread', closest_tread
#        print 'closest_dist_tread', closest_dist_tread
#        print 'on_correct_side_tread', on_correct_side_tread
        
#        attach_to_rise_not_tread = closest_dist_rise < closest_dist_tread
#        print 'attach_to_rise_not_tread', attach_to_rise_not_tread
#        
#        closest_dist = np.where(attach_to_rise_not_tread, closest_dist_rise, closest_dist_tread)
#        print 'closest_dist', closest_dist
##        on_correct_side = np.where(attach_to_rise_not_tread, on_correct_side_rise, on_correct_side_tread)
##        print 'on_correct_side', on_correct_side
#        closest_plane_ind = np.where(attach_to_rise_not_tread, closest_rise, closest_tread)
#        print 'closest_plane_ind', closest_plane_ind 
        
#        is_inlier = (on_correct_side * (closest_dist < 0.5)).astype(np.bool)
        is_inlier = (attach_plane_dist < 0.2).astype(np.bool)
        inlier_points = sample_points_roi[is_inlier]
        print 'inlier_points', inlier_points.shape[0]
        inlier_plane = attach_plane[is_inlier]
        print 'inlier_plane', inlier_plane 
        inlier_rise_not_tread = attach_to_rise_not_tread[is_inlier]    
        print 'inlier_rise_not_tread', inlier_rise_not_tread 
        inlier_points_grouped = [inlier_points[(inlier_plane==plane_ind)*b]\
                                 for plane_ind in range(plane_number)\
                                  for b in [inlier_rise_not_tread,-inlier_rise_not_tread]]
        print 'inlier_points_grouped', inlier_points_grouped
        
        inlier_weight = inlier_points.shape[0]
        print 'inlier_weight', inlier_weight 
        
#        inlier_points_grouped_rise = [inlier_points[(inlier_plane==plane_ind) * inlier_rise_not_tread]
#                                    for plane_ind in range(plane_number)]
#        print 'inlier_points_grouped_rise', inlier_points_grouped_rise
#        inlier_points_grouped_tread = [inlier_points[(inlier_plane==plane_ind) * -inlier_rise_not_tread]
#                                    for plane_ind in range(plane_number)]
#        print 'inlier_points_grouped_tread', inlier_points_grouped_tread

        
#        inlier_indices_planes = [[] fo1r i in range(plane_number)]
#        inlier_weight = 0
        
        
#        def find_closest_plane(sample_point, plane_normal):
#            if rise_front_ind > -1:
#                edgepoint_front = merged_line_3d[rise_front_ind,:3]
#                proj_front = geometry.project_to_plane(sample_point, plane_normal, edgepoint_front)
#                dist_front = linalg.norm(proj_front - edgepoint_front)
#                if dist_front < dist:
#                    dist = dist_front
#                    proj = proj_front
#                    edgepoint = edgepoint_front
#                    closer = rise_front_ind
#                
#            if rise_front_ind < len(rise_offset) - 1:
#                edgepoint_back = merged_line_3d[rise_front_ind+1,:3]
#                proj_back = geometry.project_to_plane(sample_point, plane_normal, edgepoint_back)
#                dist_back = linalg.norm(proj_back - edgepoint_back)
#                if dist_back < dist:
#                    dist = dist_back
#                    proj = proj_back
#                    edgepoint = edgepoint_back
#                    closer = rise_front_ind + 1
#            
#            if geometry.is_on_same_side_as(proj, direction_vector, edgepoint, 
#                plane_normal, np.array([0,-999,0])) or dist > 0.1:
##                print 'ignored'
#                continue
#            print sample_point, proj, dist, edgepoint, closer
        
#        for sample_ind, rise_front_ind in enumerate(front_rise):
#            dist = 9999
#            closer = None
#            sample_point = sample_points[sample_ind]
#            
#            closer = find_closest_plane_batch(sample_point, )
            
#            inlier_indices_planes[closer].append(sample_ind)
##            inlier_indices.append(sample_ind)
#            inlier_weight += 1    
            
#        inlier_points = [sample_points[inlier_indices_plane] for inlier_indices_plane in inlier_indices_planes]
#        print [len(inlier_indices_plane) for inlier_indices_plane in inlier_indices_planes]
#        for plane_ind, inlier_indices_plane in enumerate(inlier_indices_planes):
#            print plane_ind, inlier_indices_plane
        edges_plane_inlier_cloud = utility.add_to_multipointcloud_multicolor(edges_plane_cloud, 
                                                    inlier_points_grouped, plane_colors)
        utility.write_XYZRGB(edges_plane_inlier_cloud, 'edges_plane_inlier.pcd')
        
        edges_plane_inlier_box_cloud = edges_plane_inlier_cloud
#        for plane_ind in range(plane_number):
        for plane_ind in range(plane_number):
            x,d1,d2,w,h = geometry.find_boundingbox_project(inlier_points_grouped[2*plane_ind],
                                               rise_normal, edge_points[plane_ind], direction_vector)
            box_rise_one = utility.draw_box(x, d1, d2, w, h, np.array([255,255,255]))
            edges_plane_inlier_box_cloud = utility.add_to_pointcloud(edges_plane_inlier_box_cloud, box_rise_one)
            
            x,d1,d2,w,h = geometry.find_boundingbox_project(inlier_points_grouped[2*plane_ind+1],
                                               tread_normal, edge_points[plane_ind], direction_vector)
            box_tread_one = utility.draw_box(x, d1, -d2, w, h, np.array([255,255,255]))
            edges_plane_inlier_box_cloud = utility.add_to_pointcloud(edges_plane_inlier_box_cloud, box_tread_one)
        
#        utility.write_XYZRGB(edges_plane_inlier_points, 'edges_plane_inlier.pcd')
#        edges_plane_inlier_points2 = utility.add_to_multipointcloud_multicolor(edges_plane_inlier_points, 
#                                                    inlier_points_grouped_tread, tread_colors)
        utility.write_XYZRGB(edges_plane_inlier_box_cloud, 'edges_plane_inlier_box_cloud.pcd')
#        print '*******',sample_rise_ind, inlier_weight
                
sys.exit()


#normals = []
#for row in indices:
#    neighbor_x = np.array([p_list[neighbor_ind][0] for neighbor_ind in row])
#    neighbor_y = np.array([p_list[neighbor_ind][1] for neighbor_ind in row])
#    neighbor_z = np.array([p_list[neighbor_ind][2] for neighbor_ind in row])
#    P = np.column_stack((neighbor_x, neighbor_y, np.ones(neighbor_x.size)))
#    A = np.dot(P.T, P)
#    b = np.dot(P.T, neighbor_z)
#    plane_params, resid,rank,sigma = np.linalg.lstsq(A,b)
#    normal = np.array([plane_params[0],plane_params[1],-1])
#    normal = normal/linalg.norm(normal)
#    normals.append(normal)

#pv_arr = pv.to_array()
#normals = []
#for row in indices:
#    plane_params, normal = fit_plane_indices(pv_arr, row)
#    normals.append(normal)
#normals = np.array(normals)
#print normals.shape[0], 'normals computed' 

#good_plane = []
#sample_size = 8
#for iter in range(5):
#    print 'iteration', iter
#    sample_ind = [0]*sample_size
#    for i in range(sample_size):
#        sample_ind[i] = random.randint(0, pv.size-1)
#    plane_params, normal = fit_plane_indices(pv_arr, sample_ind)
#    print 'hypothesis', plane_params, normal
#    inliers = []
#    for test_ind in range(pv.size):
#        print 'testing', test_ind
#        test_dist = distance_to_plane(pv_arr, test_ind, plane_params)
#        print 'distance is ', test_dist
#        if test_dist < 0.01:
#            print 'added to inliers'
#            inliers.append(test_ind)
#    print len(inliers), 'inliers'
#    if len(inliers) > 50:
#        print 'added to good_plane'
#        good_plane.append(inliers)
#        
#print good_plane            

#    import random
#    p_sample = pcl.PointCloud()
#    p_sample.from_list([pv[random.randint(0, pv.size-1)] for i in xrange(10)])
#    indices, sqr_distances = kd.nearest_k_search_for_cloud(p_sample, 2)

#indices, sqr_distances = kd.nearest_k_search_for_cloud(pv, 100)
#mean_distances = np.mean(np.sqrt(sqr_distances), axis=1)
#print np.std(mean_distances)
#
#import matplotlib.pyplot as plt
#bins_in = np.arange(0,1,0.001)
#hist, bins = np.histogram(mean_distances, bins_in)
#width = 0.7*(bins[1]-bins[0])
#center = (bins[:-1]+bins[1:])/2
#plt.figure("mean_distances")
#plt.bar(center, hist, align = 'center', width = width)
#plt.show()

#p_inlier = pcl.PointCloud()
#p_inlier.from_array(pv.to_array()[mean_distances < 0.4])
#p_inlier.to_file("inliers.pcd")

#pv.to_file("inliers.pcd")

#    fil = pv.make_statistical_outlier_filter()
#    fil.set_mean_k (100)
#    fil.set_std_dev_mul_thresh (0.01)
#    p_fil = fil.filter()
#    print 'after outlier remover', p_fil.size
#    
#    #    p_fil2 = pcl.PointCloud()
#    #    p_fil2.from_array(p_fil.to_array()*10)
#    #    p_fil2.to_file("inliers.pcd")
    
    
    
