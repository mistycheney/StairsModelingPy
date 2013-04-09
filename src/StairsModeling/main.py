'''
Created on Apr 5, 2013

@author: yuncong
'''
import numpy as np
import os, sys
from collections import OrderedDict
import random 
import scipy.linalg as linalg


IMGPATH='/Users/yuncong/Documents/StairsModelingPy/staircase_new/'
PROJPATH='/Users/yuncong/Documents/StairsModelingPy/'

import cv2
img_id = 18
top = cv2.imread(IMGPATH+"top%d.jpg"%img_id, 0)
bottom = cv2.imread(IMGPATH+"bottom%d.jpg"%img_id, 0)
dispTop = None

os.chdir(PROJPATH)
R = np.asarray(cv2.cv.Load('extrinsics.yml', name='R'))
T = np.asarray(cv2.cv.Load('extrinsics.yml', name='T'))
R1 = np.asarray(cv2.cv.Load('extrinsics.yml', name='R1'))
R2 = np.asarray(cv2.cv.Load('extrinsics.yml', name='R2'))
P1 = np.asarray(cv2.cv.Load('extrinsics.yml', name='P1'))
P2 = np.asarray(cv2.cv.Load('extrinsics.yml', name='P2'))
Q = np.asarray(cv2.cv.Load('extrinsics.yml', name='Q'))
M1 = np.asarray(cv2.cv.Load('intrinsics.yml', name='M1'))
M2 = np.asarray(cv2.cv.Load('intrinsics.yml', name='M2'))
D1 = np.asarray(cv2.cv.Load('intrinsics.yml', name='D1'))
D2 = np.asarray(cv2.cv.Load('intrinsics.yml', name='D2'))

class ParamsTuner():
    def __init__(self, params, winname):
        self.params = params
        self.winname = winname
        cv2.namedWindow(winname, 1)
        cv2.moveWindow(winname, 0, 0)
        for k,(v,r) in params.iteritems():
            cv2.createTrackbar(k, self.winname, v, r, self.onChange)
        self.onChange(None)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def onChange(self, i):
        for k in self.params.iterkeys():
            self.params[k] = cv2.getTrackbarPos(k, self.winname), self.params[k][1]
            print k, self.params[k][0]
        print
        self.doThings()
        
    def doThings(self):
        pass
    
class EdgeTuner(ParamsTuner):
    def doThings(self):
        thresh1, thresh2,apertureSize, hough_thresh, minLineLength, maxLineGap, \
         rho_res, theta_res = [v for v,_ in self.params.itervalues()]
        global top

#        cv2.imshow('top', top)
#        cv2.imshow('Gx', Gx)
#        cv2.imshow('Gy', Gy)
#        cv2.imshow('G', (G/G.max()*255).astype(np.uint8))
        
        img = cv2.resize(top, (top.shape[1]/2,top.shape[0]/2))
#        img = cv2.equalizeHist(img)

#        img = cv2.blur(img, (3,3))
        
        Sx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        Sy = Sx.T
        Gx = cv2.filter2D(img, -1, Sx)
        Gy = cv2.filter2D(img, -1, Sy)
        G = np.sqrt(Gx.astype(np.float)**2 + Gy.astype(np.float)**2)
        cv2.imshow('G', (G/G.max()*255).astype(np.uint8))
#        cv2.imshow('Gy', Gy)

        import time
        begin = time.time()
        T = np.arctan(Gy/(0.00001+Gx))
#        print time.time() - begin
        
        begin = time.time()
        a = np.tan(22.5/180*np.pi)
        b = np.tan(67.5/180*np.pi)
        T_round = np.zeros_like(T).astype(np.int)
#        T_round = np.zeros((T.shape[0],T.shape[1],2)).astype(np.bool)
#        T_round[abs(T)<a] = [0,0]   #0
#        T_round[(T>a) * (T<b)] = [0,1]  #45
#        T_round[abs(T)>b] = [1,0]   #90
#        T_round[(T>-b) * (T<-a)] = [1,1]    #135
        T_round[abs(T)<a] = 0  #0
        T_round[(T>a) * (T<b)] = 45
        T_round[abs(T)>b] = 90   #90
        T_round[(T>-b) * (T<-a)] = 135
#        print time.time() - begin
        
        begin = time.time()
        maximum = np.zeros_like(T).astype(np.bool)
        larger_than_top = G[1:-1,1:-1] > G[:-2,1:-1]
        larger_than_bottom = G[1:-1,1:-1] > G[2:,1:-1]
        larger_than_left = G[1:-1,1:-1] > G[1:-1,:-2]
        larger_than_right = G[1:-1,1:-1] > G[1:-1,2:]
        larger_than_NE = G[1:-1,1:-1] > G[:-2,2:]
        larger_than_NW = G[1:-1,1:-1] > G[:-2,:-2]
        larger_than_SE = G[1:-1,1:-1] > G[2:,2:]
        larger_than_SW = G[1:-1,1:-1] > G[2:,:-2]
#        C0 = (T_round[1:-1,1:-1][0] == [0,0]).all(axis=1) * larger_than_top * larger_than_bottom
#        C90 = (T_round[1:-1,1:-1][0] == [1,0]).all(axis=1) * larger_than_left * larger_than_right
#        C45 = (T_round[1:-1,1:-1][0] == [0,1]).all(axis=1) * larger_than_NW * larger_than_SE
#        C135 = (T_round[1:-1,1:-1][0] == [1,1]).all(axis=1) * larger_than_NE * larger_than_SW
        C0 = (T_round[1:-1,1:-1] == 0) * larger_than_top * larger_than_bottom
        C90 = (T_round[1:-1,1:-1] == 90) * larger_than_left * larger_than_right
        C45 = (T_round[1:-1,1:-1] == 45) * larger_than_NW * larger_than_SE
        C135 = (T_round[1:-1,1:-1] == 135)* larger_than_NE * larger_than_SW
        maximum[1:-1,1:-1] = C0 + C90 + C45 + C135
        
#        for i in xrange(1, T.shape[0]-1):
#            for j in xrange(1, T.shape[1]-1):
#                maximum[i,j] = (T_round[i,j] == 0 and G[i,j] > G[i+1,j] and G[i,j] > G[i-1,j]) or \
#                (T_round[i,j] == 90 and G[i,j] > G[i,j+1] and G[i,j] > G[i,j-1]) or \
#                (T_round[i,j] == 135 and G[i,j] > G[i+1,j-1] and G[i,j] > G[i-1,j+1]) or \
#                (T_round[i,j] == 45 and G[i,j] > G[i+1,j+1] and G[i,j] > G[i-1,j-1])
#        print time.time() - begin     
        cv2.imshow('non-maximum suppression', maximum.astype(np.uint8)*255)

        begin = time.time()
        edges = np.zeros_like(T).astype(np.bool)
        larger_than_thresh2 = G > thresh2
        larger_than_thresh1 = G > thresh1
        left_larger_then_thresh2 = larger_than_thresh2[1:-1,:-2]
        right_larger_then_thresh2 = larger_than_thresh2[1:-1,2:] 
        top_larger_then_thresh2 = larger_than_thresh2[:-2,1:-1]
        bottom_larger_then_thresh2 = larger_than_thresh2[2:,1:-1]
        NE_larger_then_thresh2 = larger_than_thresh2[:-2,2:]
        NW_larger_then_thresh2 = larger_than_thresh2[:-2,:-2]
        SE_larger_then_thresh2 = larger_than_thresh2[2:,2:]
        SW_larger_then_thresh2 = larger_than_thresh2[2:,:-2]
        some_neighbor_larger_than_thresh2 = left_larger_then_thresh2 +right_larger_then_thresh2 +\
         top_larger_then_thresh2+bottom_larger_then_thresh2+NE_larger_then_thresh2+NW_larger_then_thresh2+\
         SE_larger_then_thresh2+SW_larger_then_thresh2
        edges[1:-1,1:-1] = maximum[1:-1,1:-1] * larger_than_thresh1[1:-1,1:-1] *\
                             (larger_than_thresh2[1:-1,1:-1] + some_neighbor_larger_than_thresh2)
                             
#        for i in range(T.shape[0]):
#            for j in range(T.shape[1]):
#                if maximum[i,j] and G[i,j] > thresh1:
#                    if G[i,j]>thresh2 or larger_than_thresh2[i-1:i+1,j-1:j+1].any():
#                        edges[i,j] = 255
#        print time.time() - begin

#        edges2 = cv2.Canny(Gy, thresh1, thresh2, apertureSize=apertureSize, L2gradient=True);
        cv2.imshow("edges", edges.astype(np.uint8)*255)
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

        begin = time.time()
        h,w = edges.shape
        theta = np.arange(0.0, 180.0,theta_res)
        D = np.sqrt((h - 1)**2 + (w - 1)**2)
        rho = np.arange(-D,D,rho_res)
        H = np.zeros((len(rho), len(theta))).astype(np.float)
        Coord = [[[] for col in range(len(theta))] for row in range(len(rho))]
        edges_copy = edges.copy()
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
            rho_indices = np.round((rho_vals - rho[0]) / rho_res).astype(np.int)
            Hmax = 0
            for t,r in enumerate(rho_indices):
                weight = abs(np.cos(T[y,x]-theta[t]*np.pi/180))
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
                
                gap = maxLineGap
                print 'forward'
                for offset in range(1,9999):
                    test_point = np.round(start_point + offset * follow_direction).astype(np.int)
                    if (test_point < 0).any() or (test_point >= edges.shape[::-1]).any():
                        break
                    print 'testing', test_point
                    neighbor_on_y, neighbor_on_x = np.nonzero(edges[test_point[1]-1:test_point[1]+2, test_point[0]-1:test_point[0]+2])
                    if neighbor_on_y.size > 0:
                        for i, j in zip(neighbor_on_y, neighbor_on_x):
                            neighbor_point = np.array([test_point[0]+j-1, test_point[1]+i-1], dtype=np.int)
#                            print 'neighbor_point', neighbor_point
#                            print 'current_segment', current_segment, current_segment.shape
                            if not (neighbor_point==current_segment).all(axis=1).any():
                                current_segment = np.vstack((current_segment, neighbor_point))
                                print 'append', neighbor_point, 'to current_segment'
                                gap = maxLineGap
                                print 'gap', gap
                    else:
                        gap = gap - 1
                        print 'gap', gap
                        if gap == 0: break
                        
                gap = maxLineGap
                print 'backward'
                for offset in range(1,9999):
                    test_point = np.round(start_point - offset * follow_direction).astype(np.int)
                    if (test_point < 0).any() or (test_point >= edges.shape[::-1]).any():
                        break
                    print 'testing', test_point
                    neighbor_on_y, neighbor_on_x = np.nonzero(edges[test_point[1]-1:test_point[1]+2, test_point[0]-1:test_point[0]+2])
                    if neighbor_on_y.size > 0:
                        for i, j in zip(neighbor_on_y, neighbor_on_x):
                            neighbor_point = np.array([test_point[0]+j-1, test_point[1]+i-1], dtype=np.int)
#                            print 'neighbor_point', neighbor_point
#                            print 'current_segment', current_segment, current_segment.shape
                            if not (neighbor_point==current_segment).all(axis=1).any():
                                current_segment = np.vstack((current_segment, neighbor_point))
                                print 'append', neighbor_point, 'to current_segment'
                                gap = maxLineGap
                                print 'gap', gap
                    else:
                        gap = gap - 1
                        print 'gap', gap
                        if gap == 0: break
            
                print 'current_segment len', current_segment.shape[0]
                if (current_segment.shape[0] > minLineLength):
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
                    print 'less than', minLineLength, ', reject'
                
            nz_y, nz_x = np.nonzero(edges_copy)
            print nz_y.size, 'points remain'
        
        print voting_num, 'out of', np.sum(edges.astype(np.int)), 'points voted' 
#        for seg in good_segments:
#            print seg
        print time.time() - begin



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
        
        img_color = cv2.cvtColor(img, cv2.cv.CV_GRAY2BGR);
        
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
            color = (0,0,255)
#            color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            for p in P:
                img_color[p[1],p[0]] = color
#                cv2.circle(img_color, (p[0],p[1]), 1, color)

#            cv2.line(img_color, (int(x1), int(y1)),(int(x2), int(y2)),
#                     (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 1, 8)
        print time.time() - begin
        
#        if lines is not None:
#            for x1,y1,x2,y2 in lines:
                # clip to -pi/2 to pi/2
                        
#        if lines is not None:
#            for x1,y1,x2,y2, voting_points in lines[0]:
#                # clip to -pi/2 to pi/2
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
#                cv2.line(img_color, (int(x1), int(y1)),(int(x2), int(y2)), (0, 0, 255), 1, 8)
        
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

class SGBMTuner(ParamsTuner):
    def doThings(self):
        sgbm = cv2.StereoSGBM()
        sgbm.SADWindowSize, numberOfDisparitiesMultiplier, sgbm.preFilterCap, sgbm.minDisparity, \
        sgbm.uniquenessRatio, sgbm.speckleWindowSize, sgbm.P1, sgbm.P2, \
        sgbm.speckleRange = [v for v,_ in self.params.itervalues()]
        sgbm.numberOfDisparities = numberOfDisparitiesMultiplier*16
        sgbm.disp12MaxDiff = -1
        sgbm.fullDP = False
        global dispTop
        global top
        global bottom
        global M1
        global D1
        global M2
        global D2
#        global R
#        global T
        R1, R2, P1, P2, Q,topValidRoi, bottomValidRoi = cv2.stereoRectify(M1, D1, M2, D2, 
                                (top.shape[1],top.shape[0]), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

        top_map1, top_map2 = cv2.initUndistortRectifyMap(M1, D1, R1, P1, (top.shape[1],top.shape[0]), cv2.CV_16SC2)
        bottom_map1, bottom_map2 = cv2.initUndistortRectifyMap(M2, D2, R2, P2, (bottom.shape[1],bottom.shape[0]), cv2.CV_16SC2)
        
        top_r = cv2.remap(top, top_map1, top_map2, cv2.cv.CV_INTER_LINEAR);
        bottom_r = cv2.remap(bottom, bottom_map1, bottom_map2, cv2.cv.CV_INTER_LINEAR)
        top_small = cv2.resize(top_r, (top_r.shape[1]/2,top_r.shape[0]/2))
        bottom_small = cv2.resize(bottom_r, (bottom_r.shape[1]/2,bottom_r.shape[0]/2))
        cv2.imshow('top', top_small);
        cv2.imshow('bottom', bottom_small);
        
#        top_r = cv2.equalizeHist(top_r)
        top_r = cv2.blur(top_r, (5,5))
#        bottom_r = cv2.equalizeHist(bottom_r)
        bottom_r = cv2.blur(bottom_r, (5,5))
        dispTop = sgbm.compute(top_r.T, bottom_r.T).T;
        dispTopPositive = dispTop
        dispTopPositive[dispTop<0] = 0
        disp8 = (dispTopPositive / (sgbm.numberOfDisparities * 16.) * 255).astype(np.uint8);
        disp_small = cv2.resize(disp8, (disp8.shape[1]/2, disp8.shape[0]/2));
        cv2.imshow(self.winname, disp_small);

if __name__ == '__main__':    
    edgeParams = OrderedDict([('thresh1',(73,2000)),
                              ('thresh2',(137,2000)),
                            ('apertureSize',(3,21)),
                              ('hough_thresh',(20,500)),
                              ('minLineLength',(10,500)),
                              ('maxLineGap',(20,500)),
                              ('rho',(1,50)),
                              ('theta',(1,10))])
    EdgeTuner(edgeParams, 'EdgeTuner')
    
    sys.exit()

    sgbmParams = OrderedDict([('SADWindowSize',(5,51)),
                              ('numberOfDisparitiesMultiplier',(11,1000)),
                              ('preFilterCap',(100,1000)),
                              ('minDisparity',(0,1000)),
                              ('uniquenessRatio',(3,20)),
                              ('speckleWindowSize',(0,1000)),
                              ('P1',(5300,10000)),  #300 for raw
                              ('P2',(6500,10000)),  #96564 for raw
                              ('speckleRange',(1,10))])
    SGBMTuner(sgbmParams, 'SGBMTuner')
    
    xyz = cv2.reprojectImageTo3D(dispTop, Q, handleMissingValues=True)
    print xyz.size / 3
    
    import itertools
    xyz_valid = np.array([i for i in itertools.chain(*xyz) if i[2] != 10000.]).astype(np.float32)
    print len(xyz_valid)
    
import pcl
p = pcl.PointCloud()
p_list = [(x,y,z) for x,y,z in xyz_valid if z < 10]
p.from_list(p_list)

vox = p.make_voxel_grid_filter()
vox.set_leaf_size(0.01,0.01,0.01)
pv = vox.filter()
print 'after voxel grid filter', pv.size

kd = pv.make_kdtree_flann()
indices, sqr_distances = kd.nearest_k_search_for_cloud(p, 10)
normals = []
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

pv_arr = pv.to_array()
normals = []
for row in indices:
    plane_params, normal = fit_plane_indices(pv_arr, row)
    normals.append(normal)
normals = np.array(normals)
print normals.shape[0], 'normals computed' 

good_plane = []
sample_size = 8
for iter in range(5):
    print 'iteration', iter
    sample_ind = [0]*sample_size
    for i in range(sample_size):
        sample_ind[i] = random.randint(0, pv.size-1)
    plane_params, normal = fit_plane_indices(pv_arr, sample_ind)
    print 'hypothesis', plane_params, normal
    inliers = []
    for test_ind in range(pv.size):
        print 'testing', test_ind
        test_dist = distance_to_plane(pv_arr, test_ind, plane_params)
        print 'distance is ', test_dist
        if test_dist < 0.01:
            print 'added to inliers'
            inliers.append(test_ind)
    print len(inliers), 'inliers'
    if len(inliers) > 50:
        print 'added to good_plane'
        good_plane.append(inliers)
        
print good_plane            
        

    
    
    


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

pv.to_file("inliers.pcd")

#    fil = pv.make_statistical_outlier_filter()
#    fil.set_mean_k (100)
#    fil.set_std_dev_mul_thresh (0.01)
#    p_fil = fil.filter()
#    print 'after outlier remover', p_fil.size
#    
#    #    p_fil2 = pcl.PointCloud()
#    #    p_fil2.from_array(p_fil.to_array()*10)
#    #    p_fil2.to_file("inliers.pcd")
    
    
    
