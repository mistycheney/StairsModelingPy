'''
Created on Apr 16, 2013

@author: yuncong
'''

import cPickle as pickle
import numpy as np
import os, sys
from StairsModeling import config

if __name__ == '__main__':
    os.chdir(config.PROJPATH)
    results = pickle.load(open('results.p','rb'))
    best = results[results[:,3].argmax()]
    print best
    
    import matplotlib.pyplot as plt
    bins_in = np.arange(40000,50000,10)
    hist, bins = np.histogram(results[:,3], bins_in)
#    hist, bins = np.histogram(results[:,3])
    width = 0.7*(bins[1]-bins[0])
    center = (bins[:-1]+bins[1:])/2
    plt.figure("inlier weight")
    plt.bar(center, hist, align = 'center', width = width)
    plt.show()