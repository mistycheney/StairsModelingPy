#!/bin/python

from cv2 import cv
import cv2, re
import sys, os
import time

proj_path = '/Users/yuncong/Documents/workspace/StairsPointCloud/'

output_folder = sys.argv[1]

os.chdir(proj_path + output_folder)
files = os.listdir('.')
nexisting = len([i for i in files if 'top' in i or 'left' in i])

counter = nexisting + 1

frame_list = []
frame2_list = []

vc = cv2.VideoCapture(1)
vc2 = cv2.VideoCapture(2)

rval, frame = vc.read()
rval, frame2 = vc2.read()
while rval:
    begin = time.time()
    frame_small = cv2.resize(frame, (160,90))
    frame2_small = cv2.resize(frame2, (160,90))
    cv2.imshow("top", frame_small)
    cv2.imshow("bottom", frame2_small)
    key = cv2.waitKey(20)
    rval , frame = vc.read()
    rval , frame2 = vc2.read()
    if key != -1:
        if chr(key) == 'c':
            cv2.imwrite('top'+str(counter)+".jpg",frame)
            cv2.imwrite('bottom'+str(counter)+".jpg",frame2)
            print 'image pair', counter, 'saved'
            counter = counter + 1
        if key == 27: # exit on ESC
            break


vc.release()
vc2.release()


# ---- Use c++ code instead -----
# import subprocess
# checkerboard_images = proj_path + 'calibration/left*.jpg'
# proc = subprocess.call(['python', '/Users/yuncong/OpenCV-2.4.3/samples/python2/calibrate.py',
#  '--save=calibrate_matrix.txt', '--debug='+proj_path+'/calib_debug','--square_size=22', checkerboard_images])
# os.system('python /Users/yuncong/Documents/OpenCV-2.4.2/samples/python2/calibrate.py --save=calibrate_matrix.txt --square_size=21 '+checkerboard_images)
