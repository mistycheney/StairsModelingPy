'''
Created on Apr 16, 2013

@author: yuncong
'''
import numpy as np
from collections import OrderedDict


IMGPATH='/Users/yuncong/Documents/StairsModelingPy/staircase_new/'
PROJPATH='/Users/yuncong/Documents/StairsModelingPy/'

DEFAULT_SGBM_PARAMS = OrderedDict([('SADWindowSize',(5,51)),
                              ('numberOfDisparitiesMultiplier',(11,1000)),
                              ('preFilterCap',(100,1000)),
                              ('minDisparity',(0,1000)),
                              ('uniquenessRatio',(3,20)),
                              ('speckleWindowSize',(0,1000)),
                              ('P1',(300,100000)),  #300 for raw
                              ('P2',(90000,100000)),  #90000 for raw
                              ('speckleRange',(1,10))])

DEFAULT_EDGE_PARAMS = OrderedDict([('thresh1',(73,2000)),
                              ('thresh2',(137,2000)),
                            ('apertureSize',(3,21)),
                              ('hough_thresh',(20,500)),
                              ('minLineLength',(10,500)),
                              ('maxLineGap',(100,500)),  #38
                              ('rho',(1,50)),
                              ('theta',(1,10))])

#OUTPUT_PCD = False
OUTPUT_PCD = True
#USE_ONE_NORMAL = False
USE_ONE_NORMAL = True
ONE_NORMAL = np.array([[-1.57872966e-01,  -1.88390822e-01,   9.69321940e-01]])

TUNE_LINE_EXTRACTION = False
TUNE_DISPARITY_MAP = False

LINE_REGRESSION_RANSAC_RESIDUAL_THRESH = 0.5
LINE_REGRESSION_RANSAC_INLIER_THRESH = 0.05

SAMPLE_NORMAL_NUMBER = 1000
SAMPLE_NORMAL_ANGLE_DEVIATION = 30 # +-15 degrees from projection of (0,0,1) to the normal plane of direction vector
INLIER_DISTANCE = 0.1
TEST_POINT_PERCENTAGE = 0.3 

LINE_3D_DEVIATE_HORIZONTAL_ANGLE = 20 
