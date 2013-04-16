'''
Created on Apr 8, 2013

@author: yuncong
'''

from StairsModeling import geometry, config
from StairsModeling.ParamsTuner import ParamsTuner
import cv2
import numpy as np
import random
import numpy.linalg as linalg
import time
from operator import itemgetter

class EdgeTuner(ParamsTuner):
    def __init__(self, params, winname, img):
        self.img = img
        self.do_tune = config.TUNE_LINE_EXTRACTION
        super(EdgeTuner, self).__init__(params, winname)
    
    def doThings(self):
        self.thresh1, self.thresh2, self.apertureSize, self.hough_thresh,\
         self.minLineLength, self.maxLineGap, self.rho_res, self.theta_res =\
         [v for v,_ in self.params.itervalues()]

#        self.apply_sobel()
#        self.non_maximum_suppression()
#        self.trace_with_hysteresis()
        
        begin = time.time()
        canny = cv2.Canny(self.img, self.thresh1, self.thresh2, apertureSize=self.apertureSize,
                          L2gradient=True)
        print 'Canny time', time.time() - begin
        cv2.imshow("canny", canny)
        
        begin = time.time()
        lines_list = cv2.HoughLinesP(canny.astype(np.uint8), self.rho_res, self.theta_res*np.pi/180, self.hough_thresh, 
                        minLineLength=self.minLineLength, maxLineGap=self.maxLineGap)
        print 'HoughLinesP time', time.time() - begin

#        img_lines = cv2.cvtColor(self.img, cv2.cv.CV_GRAY2BGR)

        begin = time.time()

        import sys
        if lines_list is None:
            print 'no lines detected, exit'
            sys.exit()
        lines = np.array(lines_list[0])
        line_number = lines.shape[0]
        print line_number,"lines"
#            lines_all = []
        A = np.column_stack((lines[:,0], -lines[:,1], np.ones((line_number,1))))
#        print A
        B = np.column_stack((lines[:,2], -lines[:,3], np.ones((line_number,1))))
#        print B
        line_params = np.cross(A,B)
#        print line_params
        a = line_params[:,0]
        b = line_params[:,1]
        c = line_params[:,2]
        ab = np.sqrt(a**2+b**2)
        theta = np.arctan2(-np.sign(c)*b, -np.sign(c)*a)
        rho = np.abs(c)/ab
        lines_all = np.hstack((A[:,:2], B[:,:2], np.column_stack((rho, theta))))
        horizontal_enough = abs(theta + np.pi/2) < 5*np.pi/180
        lines_horizontal = lines_all[horizontal_enough]
        print 'convert polar time', time.time() - begin

#            for i, (x1,y1,x2,y2) in enumerate(lines[0]):
#                # clip to -pi/2 to pi/2
#                print x1,y1,x2,y2
#                # from this point on, for use of hough transform, y has negative coordinate
#                line_params = np.cross(np.array([x1,-y1,1]), np.array([x2,-y2,1]))
#                a,b,c = line_params
#                
#                ab = np.sqrt(a**2+b**2)
#                if c < 0:
#                    theta = np.arctan2(b, a)
#                    rho = -c/ab
#                else:
#                    theta = np.arctan2(-b, -a)
#                    rho = c/ab
#                if abs(theta + np.pi/2) > 5*np.pi/180:
#                    continue
#                
#                print 'a,b,c', a,b,c
#                print 'theta', theta, theta*180/np.pi
#                print 'rho', rho
#                print 

#                tang = (y2-y1)/(0.00001+x2-x1)
#                if abs(tang) > 0.1:
#                    continue
#                angle = np.arctan2(y2-y1, x2-x1)
#                if angle > np.pi/2:
#                    angle = angle - np.pi
#                elif angle < -np.pi/2:
#                    angle = angle + np.pi
##                if abs(abs(angle) - np.pi/2) > np.pi/180*10:
#                if abs(angle) > np.pi/180*20:
#                    continue
                 
#                p1 = np.array([x1,y1], dtype=np.float)
#                p2 = np.array([x2,y2], dtype=np.float)
#                d = abs(np.cross(p2-p1, p1))/linalg.norm(p2-p1)
#                lines_all.append((x1,-y1,x2,-y2,rho,theta))
                
#                for p in voting_points:
#                    cv2.circle(img_color, (p[0],p[1]), 1, 
#                               (random.randint(0,255),random.randint(0,255),random.randint(0,255)))
#                cv2.line(img_lines, (int(x1), int(y1)),(int(x2), int(y2)), (0, 0, 255), 1, 8)
#                cv2.putText(img_lines, str(int(d)), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))
#                cv2.putText(img_lines, str(int(i)), (int(x1)-10, int(y1)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
#                break
        
#        def line_line_distance(line1, line2):
#            x1_l1, y1_l1, x2_l1, y2_l1, d_l1, tang_l1 = line1
#            x1_l2, y1_l2, x2_l2, y2_l2, d_l2, tang_l2 = line2
#
#            def point_to_line_distance(target_point, line_point1, line_point2):
#                x0,y0 = target_point
#                x1,y1 = line_point1
#                x2,y2 = line_point2
#                v = np.array([y2-y1, -(x2-x1)])
#                r = np.array([x1-x0, y1-y0])
#                v_norm = v/linalg.norm(v)
#                dist = abs(np.dot(v_norm, r))
#                return dist
#            
#            d_avg = (point_to_line_distance((x1_l1,y1_l1),(x1_l2,y1_l2),(x2_l2,y2_l2)) + 
#            point_to_line_distance((x2_l1,y2_l1),(x1_l2,y1_l2),(x2_l2,y2_l2)) + 
#            point_to_line_distance((x1_l2,y1_l2),(x1_l1,y1_l1),(x2_l1,y2_l1)) + 
#            point_to_line_distance((x2_l2,y2_l2),(x1_l1,y1_l1),(x2_l1,y2_l1))) / 4
#            return d_avg 
#        
#        D = np.zeros((len(lines_all), len(lines_all)))
#        for i, l1 in enumerate(lines_all):
#            for j, l2 in enumerate(lines_all):
#                D[i,j] = line_line_distance(l1,l2)
#        print D
        
#        lines_all = np.array(lines_all)

        begin = time.time()
        lines_sorted = lines_horizontal[lines_horizontal[:,4].argsort()]
        
#        lines_sort = sorted(lines_all, key=itemgetter(4))
        ds_sort = lines_sorted[:,4]
#        ds_sort = [i[4] for i in lines_sort]
        print ds_sort
        ds_sort_diff = np.array(ds_sort[1:])-np.array(ds_sort[:-1])
        group_ends = np.nonzero(abs(ds_sort_diff) > 20)[0]
        print group_ends
        
        merged_line = []
        for i in range(len(group_ends)):
            if i == 0:
                line_group = lines_sorted[:group_ends[i]+1]
            elif i == len(group_ends)-1:
                line_group = lines_sorted[group_ends[i]+1:]
            else:
                line_group = lines_sorted[group_ends[i]+1:group_ends[i+1]+1]
                
            rho_mean = np.mean(line_group[:,4])
            theta_mean = np.mean(line_group[:,5])
            print theta_mean*180/np.pi, rho_mean
            a,b,c = np.cos(theta_mean), np.sin(theta_mean), -rho_mean
#            group_number = line_group.shape[0]
#            endpoints = np.zeros((group_number,6))
            endpoints = np.vstack((line_group[:,:2], line_group[:,2:4]))
#            for x1,y1,x2,y2,rho,theta in line_group:
#                if endpoints is None:
#                    endpoints = np.array([[x1,y1],[x2,y2]])
#                else:
#                    endpoints = np.vstack((endpoints, np.array([[x1,y1],[x2,y2]])))
#            print a/c, b/c
            endpoints_proj = geometry.project_point_to_line(endpoints, (a/c, b/c))
            e1 = endpoints_proj[np.argmin(endpoints_proj[:,0])]
            e2 = endpoints_proj[np.argmax(endpoints_proj[:,0])]
            print e1,e2
            merged_line.append((rho_mean, theta_mean, e1[0], -e1[1], e2[0], -e2[1]))
            
#            for (x1, y1, x2, y2, rho, theta) in line_group:
#                cv2.line(img_lines, (int(x1), int(y1)),(int(x2), int(y2)), color, 1, 8)
        
        merged_line = np.array(merged_line)
        merged_line = merged_line[merged_line[:,0].argsort()[::-1]]
        
        print 'merging line time', time.time() - begin
        
        print 'merged_line'
        img_lines = cv2.cvtColor(self.img, cv2.cv.CV_GRAY2BGR)
        img_lines = cv2.resize(img_lines, (img_lines.shape[1]/2,img_lines.shape[0]/2)) 
        for i, (rho,theta,x1,y1,x2,y2) in enumerate(merged_line):
#            print i, i[1]*180/np.pi
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))       
            cv2.line(img_lines, (int(x1/2), int(y1/2)),(int(x2/2), int(y2/2)), color, 1, 8)
            cv2.putText(img_lines, str(int(i)), (int(x1/2)-10, int(y1/2)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0)) 
        cv2.imshow(self.winname, img_lines)

        # adhoc filter
        self.merged_line = merged_line[:-2]
        
                
#        self.hough_transform()

#        cv2.imshow('top', top)
#        cv2.imshow('Gx', Gx)
#        cv2.imshow('Gy', Gy)
#        cv2.imshow('G', (G/G.max()*255).astype(np.uint8))
        
    def apply_sobel(self):
#        img = cv2.equalizeHist(img)
        img = cv2.blur(self.img, (5,5))
        Sx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        Sy = Sx.T
        self.Gx = cv2.filter2D(img, -1, Sx)
        self.Gy = cv2.filter2D(img, -1, Sy)
        self.G = np.sqrt(self.Gx.astype(np.float)**2 + self.Gy.astype(np.float)**2)
        cv2.imshow('G', (self.G/self.G.max()*255).astype(np.uint8))
        self.T = np.arctan(self.Gy/(0.00001+self.Gx))
#        cv2.imshow('Gy', Gy)
    
    def non_maximum_suppression(self):

        a = np.tan(22.5/180*np.pi)
        b = np.tan(67.5/180*np.pi)
        T_round = np.zeros_like(self.T).astype(np.int)
        T_round[abs(self.T)<a] = 0  #0
        T_round[(self.T>a) * (self.T<b)] = 45
        T_round[abs(self.T)>b] = 90   #90
        T_round[(self.T>-b) * (self.T<-a)] = 135
        
        self.maximum = np.zeros_like(self.T).astype(np.bool)
        
        larger_than_top = self.G[1:-1,1:-1] > self.G[:-2,1:-1]
        larger_than_bottom = self.G[1:-1,1:-1] > self.G[2:,1:-1]
        larger_than_left = self.G[1:-1,1:-1] > self.G[1:-1,:-2]
        larger_than_right = self.G[1:-1,1:-1] > self.G[1:-1,2:]
        larger_than_NE = self.G[1:-1,1:-1] > self.G[:-2,2:]
        larger_than_NW = self.G[1:-1,1:-1] > self.G[:-2,:-2]
        larger_than_SE = self.G[1:-1,1:-1] > self.G[2:,2:]
        larger_than_SW = self.G[1:-1,1:-1] > self.G[2:,:-2]
        C0 = (T_round[1:-1,1:-1] == 0) * larger_than_left * larger_than_right
        C90 = (T_round[1:-1,1:-1] == 90) * larger_than_top * larger_than_bottom
        C45 = (T_round[1:-1,1:-1] == 45) * larger_than_NE * larger_than_SW
        C135 = (T_round[1:-1,1:-1] == 135)* larger_than_NW * larger_than_SE
        self.maximum[1:-1,1:-1] = C0 + C90 + C45 + C135
        
#        for i in xrange(1, T.shape[0]-1):
#            for j in xrange(1, T.shape[1]-1):
#                maximum[i,j] = (T_round[i,j] == 0 and G[i,j] > G[i+1,j] and G[i,j] > G[i-1,j]) or \
#                (T_round[i,j] == 90 and G[i,j] > G[i,j+1] and G[i,j] > G[i,j-1]) or \
#                (T_round[i,j] == 135 and G[i,j] > G[i+1,j-1] and G[i,j] > G[i-1,j+1]) or \
#                (T_round[i,j] == 45 and G[i,j] > G[i+1,j+1] and G[i,j] > G[i-1,j-1])
#        print time.time() - begin     
        cv2.imshow('non-maximum suppression', self.maximum.astype(np.uint8)*255)

    def trace_with_hysteresis(self):
        
        self.G_thin_edge = self.G * self.maximum
        self.G_image = (self.G_thin_edge/self.G_thin_edge.max()*255).astype(np.uint8)
        edges_color = cv2.cvtColor(self.G_image, cv2.cv.CV_GRAY2BGR)
        edges_color[(self.G_thin_edge > self.thresh2) * (self.G_thin_edge > 0)] = (255,255,255)
        edges_color[(self.G_thin_edge < self.thresh1) * (self.G_thin_edge > 0)] = (100,100,100)
        edges_color[(self.G_thin_edge <= self.thresh2) * (self.G_thin_edge >= self.thresh1)] = (255,0,0)
        cv2.imshow("edges_color", edges_color.astype(np.uint8))
        
        cnts, hiers = cv2.findContours(self.maximum.astype(np.uint8), cv2.cv.CV_RETR_LIST, cv2.cv.CV_CHAIN_APPROX_NONE)
        
#        for c in cnts:
#            is_strong = False
#            for x,y in np.squeeze(c,axis=1):
#                print "test", x, y
##                mask = edges_color==(255,255,255)
##                cv2.floodFill(edges_color, mask, (x,y), (255,255,255))
#                if self.G_thin_edge[y,x] > self.thresh2:
#                    is_strong = True
#                    break
#            if is_strong:
#                for x,y in np.squeeze(c,axis=1):
#                    if not (edges_color[y,x] == (255,255,255)).all():
##                        print x,y,'red'
#                        edges_color[y,x] = (0,0,255)
        
        cv2.imshow("BLOB", edges_color.astype(np.uint8))

        

#        self.edges = np.zeros_like(self.T).astype(np.bool)
#        larger_than_thresh2 = self.G > self.thresh2
#        larger_than_thresh1 = self.G > self.thresh1
#        left_larger_then_thresh2 = larger_than_thresh2[1:-1,:-2]
#        right_larger_then_thresh2 = larger_than_thresh2[1:-1,2:] 
#        top_larger_then_thresh2 = larger_than_thresh2[:-2,1:-1]
#        bottom_larger_then_thresh2 = larger_than_thresh2[2:,1:-1]
#        NE_larger_then_thresh2 = larger_than_thresh2[:-2,2:]
#        NW_larger_then_thresh2 = larger_than_thresh2[:-2,:-2]
#        SE_larger_then_thresh2 = larger_than_thresh2[2:,2:]
#        SW_larger_then_thresh2 = larger_than_thresh2[2:,:-2]
#        some_neighbor_larger_than_thresh2 = left_larger_then_thresh2 +right_larger_then_thresh2 +\
#         top_larger_then_thresh2+bottom_larger_then_thresh2+NE_larger_then_thresh2+NW_larger_then_thresh2+\
#         SE_larger_then_thresh2+SW_larger_then_thresh2
#        self.edges[1:-1,1:-1] = self.maximum[1:-1,1:-1] * larger_than_thresh1[1:-1,1:-1] *\
#                             (larger_than_thresh2[1:-1,1:-1] + some_neighbor_larger_than_thresh2)
        
        
        
        
#        for i in range(T.shape[0]):
#            for j in range(T.shape[1]):
#                if maximum[i,j] and G[i,j] > thresh1:
#                    if G[i,j]>thresh2 or larger_than_thresh2[i-1:i+1,j-1:j+1].any():
#                        edges[i,j] = 255
#        print time.time() - begin

#        edges2 = cv2.Canny(Gy, thresh1, thresh2, apertureSize=apertureSize, L2gradient=True);
#        cv2.imshow("edges", self.edges.astype(np.uint8)*255)
#        return
        
#        lines = cv2.HoughLinesP(edges, rho, theta, max(10, hough_thresh),
#                minLineLength=max(10, minLineLength), maxLineGap=max(10, maxLineGap));
#        begin = time.time()
##        def hough_transform(img_bin, theta_res=1, rho_res=1):
#        h,w = edges.shape
#        theta = np.arange(0.0, 180.0,theta_res)
#        D = np.sqrt((h - 1)**2 + (w - 1)**2)
#        rho = np.arange(-D,D,rho_res)
#        H = np.zeros((len(rho), len(theta))).astype(np.float)
#        Coord = [[[] for col in range(len(theta))] for row in range(len(rho))]
#        nz_y, nz_x = np.nonzero(edges)
#        for y,x in zip(nz_y, nz_x):
#            rho_vals = x*np.cos(theta*np.pi/180.0) + y*np.sin(theta*np.pi/180)
#            rho_indices = np.round((rho_vals - rho[0]) / rho_res).astype(np.int)
##            print rho_indices
#            for t,r in enumerate(rho_indices):
#                weight = abs(np.cos(T[y,x]-theta[t]*np.pi/180))
#                if weight > 0.8:
#                    H[r,t] += weight
#                    Coord[r][t].append([x,y])
#        print time.time() - begin


    def hough_transform(self):
        h,w = self.edges.shape
        theta = np.arange(0.0, 180.0,self.theta_res)
        D = np.sqrt((h - 1)**2 + (w - 1)**2)
        rho = np.arange(-D,D,self.rho_res)
        H = np.zeros((len(rho), len(theta))).astype(np.float)
        Coord = [[[] for col in range(len(theta))] for row in range(len(rho))]
        edges_copy = self.edges.copy()
        good_segments = []
        nz_y, nz_x = np.nonzero(edges_copy)
        voting_num = 0
        print nz_y.size, 'points remain'
        while nz_y.size > 0:
#        for y,x in zip(nz_y, nz_x):
            voting_num += 1
            i = random.randint(0, nz_y.size-1)
            y, x = nz_y[i], nz_x[i]
            start_point = np.array([x,y], dtype=np.int)
            edges_copy[y,x] = 0
            rho_vals = x*np.cos(theta*np.pi/180.0) + y*np.sin(theta*np.pi/180)
            rho_indices = np.round((rho_vals - rho[0]) / self.rho_res).astype(np.int)
            Hmax = 0
            for t,r in enumerate(rho_indices):
                weight = abs(np.cos(self.T[y,x]-theta[t]*np.pi/180))
                if weight > 0.8:
                    H[r,t] += weight
#                    print "point [", x,y, "] voted H[%d,%d]"%(r,t), weight
                    Coord[r][t].append([x,y,weight])
                    if H[r,t] > Hmax:
                        rmax = r
                        tmax = t
                        Hmax = H[r,t]
            if Hmax < 0.9:
                print 'eliminated', start_point
#                edges_copy = edges.copy()
#                nz_y, nz_x = np.nonzero(edges_copy)
                continue
            else:
                follow_direction = np.array([np.sin(tmax*np.pi/180),np.cos(tmax*np.pi/180)], dtype=np.float)
                print "potential line", rho[rmax], theta[tmax], follow_direction
#                step = 1
                print 'start_point', start_point
                current_segment = np.array([start_point])
                
                gap = self.maxLineGap
                print 'forward'
                for offset in range(1,9999):
                    test_point = np.round(start_point + offset * follow_direction).astype(np.int)
                    if (test_point < 0).any() or (test_point >= self.edges.shape[::-1]).any():
                        break
                    print 'testing', test_point
                    neighbor_on_y, neighbor_on_x = np.nonzero(self.edges[test_point[1]-1:test_point[1]+2, test_point[0]-1:test_point[0]+2])
                    if neighbor_on_y.size > 0:
                        for i, j in zip(neighbor_on_y, neighbor_on_x):
                            neighbor_point = np.array([test_point[0]+j-1, test_point[1]+i-1], dtype=np.int)
#                            print 'neighbor_point', neighbor_point
#                            print 'current_segment', current_segment, current_segment.shape
                            if not (neighbor_point==current_segment).all(axis=1).any():
                                current_segment = np.vstack((current_segment, neighbor_point))
                                print 'append', neighbor_point, 'to current_segment'
                                gap = self.maxLineGap
                                print 'gap', gap
                    else:
                        gap = gap - 1
                        print 'gap', gap
                        if gap == 0: break
                        
                gap = self.maxLineGap
                print 'backward'
                for offset in range(1,9999):
                    test_point = np.round(start_point - offset * follow_direction).astype(np.int)
                    if (test_point < 0).any() or (test_point >= self.edges.shape[::-1]).any():
                        break
                    print 'testing', test_point
                    neighbor_on_y, neighbor_on_x = np.nonzero(self.edges[test_point[1]-1:test_point[1]+2, test_point[0]-1:test_point[0]+2])
                    if neighbor_on_y.size > 0:
                        for i, j in zip(neighbor_on_y, neighbor_on_x):
                            neighbor_point = np.array([test_point[0]+j-1, test_point[1]+i-1], dtype=np.int)
#                            print 'neighbor_point', neighbor_point
#                            print 'current_segment', current_segment, current_segment.shape
                            if not (neighbor_point==current_segment).all(axis=1).any():
                                current_segment = np.vstack((current_segment, neighbor_point))
                                print 'append', neighbor_point, 'to current_segment'
                                gap = self.maxLineGap
                                print 'gap', gap
                    else:
                        gap = gap - 1
                        print 'gap', gap
                        if gap == 0: break
            
                print 'current_segment len', current_segment.shape[0]
                if (current_segment.shape[0] > self.minLineLength):
#                    edges_copy[current_segment[:,1],current_segment[:,0]] = 0
                    for x,y in current_segment:
                        edges_copy[y,x] = 0
                        for x_voted,y_voted,w in Coord[rmax][tmax]:
                            if x==x_voted and y==y_voted:
                                print "revoke point [", x,y, "] vote", w, 'on H[%d,%d]'%(rmax,tmax)
                                H[rmax,tmax] -= w
                    good_segments.append(current_segment)
                    print 'added to good_segments'
                else:
                    print 'less than', self.minLineLength, ', reject'
                
            nz_y, nz_x = np.nonzero(edges_copy)
            print nz_y.size, 'points remain'
        
        print voting_num, 'out of', np.sum(self.edges.astype(np.int)), 'points voted' 
#        for seg in good_segments:
#            print seg



#        begin = time.time()
#        print "H.max()", H.max()
#        lines = [[]]
#        H[H < hough_thresh] = 0        
        
#        import scipy.linalg as linalg
#        for r,t in zip(*np.nonzero(H)):
#            print t,r,Coord[r][t]
#            P = np.atleast_2d(Coord[r][t])
#            X = P[:,0]
#            Y = P[:,1]
#            X_homo = np.column_stack((X, np.ones((P.shape[0],1))))
#            d, _,_,_ = np.array(linalg.lstsq(X_homo,Y))
#            a = d[0]
#            b = d[1]
#            Xp = (P[:,0]-a*(Y-b))/(1+a**2)
#            x1 = Xp.min()
#            x2 = Xp.max()
#            y1 = a*x1+b
#            y2 = a*x2+b
#            lines[0].append([x1, y1, x2, y2, Coord[r][t]])
#        print time.time() - begin
        
#        import matplotlib.pyplot as plt
#        from mpl_toolkits.mplot3d import Axes3D
#        from matplotlib import cm
#        from matplotlib.ticker import LinearLocator, FormatStrFormatter
#        fig = plt.figure()
#        ax = fig.gca(projection='3d')
#        X = theta
#        Y = rho
#        X, Y = np.meshgrid(X, Y)
#        surf = ax.plot_surface(X, Y, H, rstride=1, cstride=1, cmap=cm.coolwarm,
#                linewidth=0, antialiased=False)
###        ax.set_zlim(-1000, 1000)
#        ax.zaxis.set_major_locator(LinearLocator(10))
#        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#        ax.set_xlabel("theta")
#        ax.set_ylabel("rho")
#        ax.set_zlabel("vote")
        
#        fig.colorbar(surf, shrink=0.5, aspect=5)
#        
#        plt.show()
        cv2.imshow("Hough transform",(H.astype(np.float)/H.max()*255).astype(np.uint8))
        
        img_color = cv2.cvtColor(self.img, cv2.cv.CV_GRAY2BGR);
        
        lines = []
        for P in good_segments:
            X = P[:,0]
            Y = P[:,1]
            X_homo = np.column_stack((X, np.ones((P.shape[0],1))))
            d, _,_,_ = np.array(linalg.lstsq(X_homo,Y))
            a = d[0]
            b = d[1]
            Xp = (P[:,0]-a*(Y-b))/(1+a**2)
            x1 = Xp.min()
            x2 = Xp.max()
            y1 = a*x1+b
            y2 = a*x2+b
            lines.append([x1, y1, x2, y2])
            
#            angle = np.arctan2(y2-y1, x2-x1)
#            if angle > np.pi/2:
#                angle = angle - np.pi
#            elif angle < -np.pi/2:
#                angle = angle + np.pi
##                if abs(abs(angle) - np.pi/2) > np.pi/180*10:
#            if abs(angle) > np.pi/180*5:
#                continue
            color = (0,0,255 )
#            color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            for p in P:
                img_color[p[1],p[0]] = color
#                cv2.circle(img_color, (p[0],p[1]), 1, color)

#            cv2.line(img_color, (int(x1), int(y1)),(int(x2), int(y2)),
#                     (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 1, 8)
        
#        if lines is not None:
#            for x1,y1,x2,y2 in lines:
                # clip to -pi/2 to pi/2
                        

        
#        if len(good_segments) > 0:
#            for segment in good_segments:
#                for i,j in segment:
#                    img_color[j,i] = (0,0,255)
#                # clip to -pi/2 to pi/2
#                S = np.array(segment)
#                s1 = np.argmin(S[:,0], 0)
#                s2 = np.argmax(S[:,0], 0)
#                x1 = S[s1,0]
#                y1 = S[s1,1]
#                x2 = S[s2,0]
#                y2 = S[s2,1]
#                angle = np.arctan2(y2-y1, x2-x1)
#                if angle > np.pi/2:
#                    angle = angle - np.pi
#                elif angle < -np.pi/2:
#                    angle = angle + np.pi
##                if abs(abs(angle) - np.pi/2) > np.pi/180*10:
#                if abs(angle) > np.pi/180*5:
#                    continue
##                for p in voting_points:
##                    cv2.circle(img_color, (p[0],p[1]), 1, 
##                               (random.randint(0,255),random.randint(0,255),random.randint(0,255)))
#                cv2.line(img_color, (int(x1), int(y1)),(int(x2), int(y2)), (0, 0, 255), 1, 8);
        cv2.imshow(self.winname, img_color);
