from StairsModeling.ParamsTuner import ParamsTuner
import cv2
import numpy as np
import random
import numpy.linalg as linalg

class SGBMTuner(ParamsTuner):
    def __init__(self, params, winname, top, bottom):
        self.top = top
        self.bottom = bottom

#        IMGPATH='/Users/yuncong/Documents/StairsModelingPy/staircase_new/'
        PROJPATH='/Users/yuncong/Documents/StairsModelingPy/'
        extrinsic_filepath = PROJPATH + 'extrinsics.yml'
        intrinsic_filepath = PROJPATH + 'intrinsics.yml'
        self.R = np.asarray(cv2.cv.Load(extrinsic_filepath, name='R'))
        self.T = np.asarray(cv2.cv.Load(extrinsic_filepath, name='T'))
        self.R1 = np.asarray(cv2.cv.Load(extrinsic_filepath, name='R1'))
        self.R2 = np.asarray(cv2.cv.Load(extrinsic_filepath, name='R2'))
        self.P1 = np.asarray(cv2.cv.Load(extrinsic_filepath, name='P1'))
        self.P2 = np.asarray(cv2.cv.Load(extrinsic_filepath, name='P2'))
        self.Q = np.asarray(cv2.cv.Load(extrinsic_filepath, name='Q'))
        self.M1 = np.asarray(cv2.cv.Load(intrinsic_filepath, name='M1'))
        self.M2 = np.asarray(cv2.cv.Load(intrinsic_filepath, name='M2'))
        self.D1 = np.asarray(cv2.cv.Load(intrinsic_filepath, name='D1'))
        self.D2 = np.asarray(cv2.cv.Load(intrinsic_filepath, name='D2'))
        
        super(SGBMTuner, self).__init__(params, winname)
    
    def doThings(self):
        sgbm = cv2.StereoSGBM()
        sgbm.SADWindowSize, numberOfDisparitiesMultiplier, sgbm.preFilterCap, sgbm.minDisparity, \
        sgbm.uniquenessRatio, sgbm.speckleWindowSize, sgbm.P1, sgbm.P2, \
        sgbm.speckleRange = [v for v,_ in self.params.itervalues()]
        sgbm.numberOfDisparities = numberOfDisparitiesMultiplier*16
        sgbm.disp12MaxDiff = -1
        sgbm.fullDP = False
        R1, R2, P1, P2, Q, topValidRoi, bottomValidRoi = cv2.stereoRectify(self.M1, self.D1, self.M2, self.D2, 
                                (self.top.shape[1],self.top.shape[0]), self.R, self.T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

        top_map1, top_map2 = cv2.initUndistortRectifyMap(self.M1, self.D1, R1, P1, 
                                                         (self.top.shape[1],self.top.shape[0]), cv2.CV_16SC2)
        bottom_map1, bottom_map2 = cv2.initUndistortRectifyMap(self.M2, self.D2, R2, P2, 
                                                               (self.bottom.shape[1], self.bottom.shape[0]), cv2.CV_16SC2)
        
        self.top_r = cv2.remap(self.top, top_map1, top_map2, cv2.cv.CV_INTER_LINEAR);
        self.bottom_r = cv2.remap(self.bottom, bottom_map1, bottom_map2, cv2.cv.CV_INTER_LINEAR)
        top_small = cv2.resize(self.top_r, (self.top_r.shape[1]/2,self.top_r.shape[0]/2))
        bottom_small = cv2.resize(self.bottom_r, (self.bottom_r.shape[1]/2,self.bottom_r.shape[0]/2))
        cv2.imshow('top', top_small);
        cv2.imshow('bottom', bottom_small);
        
#        top_r = cv2.equalizeHist(top_r)
        top_r = cv2.blur(self.top_r, (5,5))
#        bottom_r = cv2.equalizeHist(bottom_r)
        bottom_r = cv2.blur(self.bottom_r, (5,5))
        dispTop = sgbm.compute(top_r.T, bottom_r.T).T;
        dispTopPositive = dispTop
        dispTopPositive[dispTop<0] = 0
        disp8 = (dispTopPositive / (sgbm.numberOfDisparities * 16.) * 255).astype(np.uint8);
        disp_small = cv2.resize(disp8, (disp8.shape[1]/2, disp8.shape[0]/2));
        cv2.imshow(self.winname, disp_small);
        
        self.disp8 = disp8
        self.xyz = cv2.reprojectImageTo3D(dispTop, Q, handleMissingValues=True)
#        self.xyzrgb = np.zeros((self.xyz.shape[0],self.xyz.shape[1],4))
        
#        import struct
#        def color_to_float(color):
#            if color.size == 1:
#                color = [color]*3
#            rgb = (color[2] << 16 | color[1] << 8 | color[0]);
#            rgb_hex = hex(rgb)[2:-1]
#            s = '0'*(8-len(rgb_hex)) + rgb_hex.capitalize()
##            print color, rgb, hex(rgb)
#            rgb_float = struct.unpack('!f', s.decode('hex'))[0]
##            print rgb_float
#            return rgb_float
        
#        for i in range(self.xyz.shape[0]):
#            for j in range(self.xyz.shape[1]):
#                self.xyzrgb[i,j] = np.append(self.xyz[i,j], color_to_float(self.top[i,j])) 
        