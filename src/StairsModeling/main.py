'''
Created on Apr 5, 2013

@author: yuncong
'''

import os, sys
import numpy as np
import random 
import numpy.linalg as linalg
from StairsModeling import Edge, StereoMatch, utility, geometry, config
import itertools
import pcl
import time
import cv2

img_id = 18
top = cv2.imread(config.IMGPATH+"top%d.jpg"%img_id, 0)
#top = cv2.resize(top_o, (top_o.shape[1]/2, top_o.shape[0]/2))
bottom = cv2.imread(config.IMGPATH+"bottom%d.jpg"%img_id, 0)
#bottom = cv2.resize(bottom_o, (bottom_o.shape[1]/2, bottom_o.shape[0]/2))

if __name__ == '__main__':
    os.chdir(config.PROJPATH)
    
    sgbmParams = config.DEFAULT_SGBM_PARAMS
    sgbm = StereoMatch.SGBMTuner(sgbmParams, 'SGBMTuner', top, bottom)
        
    xyz_valid = np.array([i for i in itertools.chain(*sgbm.xyz) if i[2] != 10000.]).astype(np.float32)
    print xyz_valid.shape
    
#    xyzrgb_valid = np.array([i for i in itertools.chain(*sgbm.xyzrgb) if i[2] != 10000.]).astype(np.float32)
#    print xyzrgb_valid.shape

    xyz = sgbm.xyz
    
    edgeParams = config.DEFAULT_EDGE_PARAMS
    edge = Edge.EdgeTuner(edgeParams, 'EdgeTuner', sgbm.top_r)
    
    line_points = None
#    disp_color = cv2.cvtColor(sgbm.disp8, cv2.cv.CV_GRAY2BGR)
#    step = 0.01
    merged_line_3d = []
    
    begin = time.time()
    for i, (rho,theta,x1,y1,x2,y2) in enumerate(edge.merged_line):
        x1p, y1p, x2p, y2p = int(x1), int(y1), int(x2), int(y2)
        p1 = np.array([x1p, y1p])
        p2 = np.array([x2p, y2p]) 
#        cv2.line(disp_color, (x1p, y1p), (x2p, y2p), (0,0,255))
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
        
        if np.dot(direction_vector, np.array([1,0,0])) > np.cos(config.LINE_3D_DEVIATE_HORIZONTAL_ANGLE*np.pi/180):
#        if np.dot(direction_vector, np.array([1,0,0])) > 0.95:
            print 'added'
            merged_line_3d.append(np.hstack((center_point, direction_vector)))

        if line_points is None:
            line_points = Mhat
        else:
#            line_points = np.vstack((line_points, Mhat,interm_points_3d_valid))
            line_points = np.vstack((line_points, Mhat))
    
    print 'PCA edge line time', time.time() - begin
    
#    cv2.imshow("disp_color", disp_color)
#    cv2.waitKey()
        
#    xyz_valid = np.array([i for i in itertools.chain(*xyz) if i[2] != 10000.]).astype(np.float32)
#    edge_array = np.vstack((np.array(line_points), xyz_valid))    
#    xyzrgb_valid = [color_to_float(c) for c in top]

    p = pcl.PointCloud()
    p.from_array(xyz_valid)
    vox = p.make_voxel_grid_filter()
    vox.set_leaf_size(0.01,0.01,0.01)
    pv = vox.filter()
    xyz_downsampled = pv.to_array()
    point_number = xyz_downsampled.shape[0]
    print 'after voxel grid filter', point_number
    
#    xyz_normals = geometry.compute_cloud_normals(xyz_downsampled, k=50)
#    normals_cloud = utility.draw_normals(xyz_downsampled, xyz_normals)
#    xyz_cloud = utility.paint_pointcloud(xyz_downsampled, np.array([0,0,255]))
#    xyz_normals_cloud = utility.add_to_pointcloud_color(xyz_cloud, normals_cloud, np.array([255,255,255]))
#    utility.write_XYZRGB(xyz_normals_cloud, 'xyz_normals_cloud.pcd')
    
    merged_line_3d = np.array(merged_line_3d)
    plane_number = merged_line_3d.shape[0]
    direction_vector = np.mean(merged_line_3d[:,3:], axis=0)
    direction_vector = direction_vector/linalg.norm(direction_vector)
    print 'direction_vector', direction_vector
    edge_points = merged_line_3d[:,:3]

    if config.USE_ONE_NORMAL:
        sample_rise_normal = config.ONE_NORMAL
    else:
        sample_rise_normal = geometry.sample_vector_normal_to_vector(direction_vector, config.SAMPLE_NORMAL_NUMBER)         
#    rise_colors = np.random.randint(0,255,(plane_number,3))
#    tread_colors = np.random.randint(0,255,(plane_number,3))
    plane_colors =  np.random.randint(0,255,(2*plane_number,3))
    plane_colors[0] = np.array([255,0,0])
    plane_colors[1] = np.array([0,255,0])

    results = np.hstack((sample_rise_normal, np.zeros((sample_rise_normal.shape[0],1))))
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
        
        if config.OUTPUT_PCD:
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
        
        sample_indices = np.random.randint(0, point_number, int(point_number*config.TEST_POINT_PERCENTAGE))
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
#        print farther_than_rise
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
#        print farther_than_rise
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
#            print 'edge',edge_ind, edge_point
            is_comparing_this_plane = planes_to_compare_roi == edge_ind 
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

        is_inlier = (attach_plane_dist < config.INLIER_DISTANCE).astype(np.bool)
        inlier_points = sample_points_roi[is_inlier]
        print 'inlier_points', inlier_points.shape[0]
        inlier_plane = attach_plane[is_inlier]
        print 'inlier_plane', inlier_plane 
        inlier_rise_not_tread = attach_to_rise_not_tread[is_inlier]    
        print 'inlier_rise_not_tread', inlier_rise_not_tread 
        inlier_points_grouped = [inlier_points[(inlier_plane==plane_ind)*b]\
                                 for plane_ind in range(plane_number)\
                                  for b in [inlier_rise_not_tread,-inlier_rise_not_tread]]
#        print 'inlier_points_grouped', inlier_points_grouped
        
        inlier_weight = inlier_points.shape[0]
        print '***** rise_normal', rise_normal, 'inlier_weight', inlier_weight
        results[sample_rise_ind,3] = inlier_weight
        
        if config.OUTPUT_PCD:
        
            edges_plane_all_cloud = utility.add_to_pointcloud_color(edges_plane_cloud, xyz_downsampled, np.array([0,0,255])) 
    #        utility.write_XYZRGB(edges_plane_all_cloud, 'edges_plane_all_cloud.pcd')

            box_cloud = None
            for plane_ind in range(plane_number):
                x,d1,d2,w,h = geometry.find_boundingbox_project(inlier_points_grouped[2*plane_ind],
                                                   rise_normal, edge_points[plane_ind], direction_vector)
                box_rise_one = utility.draw_box(x, d1, d2, w, h, np.array([255,255,255]))
                box_cloud = utility.add_to_pointcloud(box_cloud, box_rise_one)
                
                x,d1,d2,w,h = geometry.find_boundingbox_project(inlier_points_grouped[2*plane_ind+1],
                                                   tread_normal, edge_points[plane_ind], direction_vector)
                box_tread_one = utility.draw_box(x, d1, -d2, w, h, np.array([255,255,255]))
                box_cloud = utility.add_to_pointcloud(box_cloud, box_tread_one)
    
            edges_plane_inlier_cloud = utility.add_to_multipointcloud_multicolor(edges_plane_cloud, 
                                                        inlier_points_grouped, plane_colors)
            edges_plane_inlier_box_cloud = utility.add_to_pointcloud(edges_plane_inlier_cloud, box_cloud)
            utility.write_XYZRGB(edges_plane_inlier_box_cloud, 'edges_plane_inlier_box_cloud.pcd')
            
    #        inlier_normals = geometry.compute_cloud_normals(inlier_points, k=50)
    #        normals_cloud = utility.draw_normals(inlier_points, inlier_normals)
    #        inlier_cloud = utility.paint_pointcloud(inlier_points, np.array([0,0,255]))
    #        inlier_normals_cloud = utility.add_to_pointcloud_color(inlier_cloud, normals_cloud, np.array([255,255,255]))
    #        utility.write_XYZRGB(inlier_normals_cloud, 'inlier_normals_cloud.pcd')
    #        sys.exit()
                
            edges_plane_all_box_cloud = utility.add_to_pointcloud(edges_plane_all_cloud, box_cloud)
            utility.write_XYZRGB(edges_plane_all_box_cloud, 'edges_plane_all_box_cloud.pcd')
    
    import cPickle as pickle
    pickle.dump(results, open('results.p','wb'))
        
sys.exit()

def statistical_outlier_removal(points, kd=None):
    cloud = pcl.PointCloud()
    cloud.from_array(points)
    if kd is None:
        kd = cloud.make_kdtree_flann()
    indices, sqr_distances = kd.nearest_k_search_for_cloud(cloud, 100)
    mean_distances = np.mean(np.sqrt(sqr_distances), axis=1)
    print np.std(mean_distances)
    
    import matplotlib.pyplot as plt
    bins_in = np.arange(0,1,0.001)
    hist, bins = np.histogram(mean_distances, bins_in)
    width = 0.7*(bins[1]-bins[0])
    center = (bins[:-1]+bins[1:])/2
    plt.figure("mean_distances")
    plt.bar(center, hist, align = 'center', width = width)
    plt.show()
    
#    p_inlier = pcl.PointCloud()
#    p_inlier.from_array(pv.to_array()[mean_distances < 0.4])
#    p_inlier.to_file("inliers.pcd")

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
    
    
    
