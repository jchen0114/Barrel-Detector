import os, cv2
from skimage.measure import label, regionprops
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

barrel, back = 0, 0
prior_back, prior_barrel = 0.75, 0.25
prior_new, prior_otherblue = 0.65, 0.35
result = np.zeros((800,1200))
result1 = np.zeros((800,1200))

blist,backlist = [], []
blueother, other = [], []

for j in range(1,47):
    im = cv2.imread('C:/Users/justi/Desktop/win19/276A/ECE276A_HW1/trainset/{}.png'.format(j))
    x = np.load('C:/Users/justi/Desktop/win19/276A/HW1/label/label{}.npy'.format(j))
    for k in range(0,800):
        for l in range(0,1200):
            if x[k][l] == True:
                blist.append(im[k][l])
            else:
                backlist.append(im[k][l]) 
                        
    otherblue = np.load('C:/Users/justi/Desktop/win19/276A/HW1/otherblue/otherblue{}.npy'.format(j))
    for a in range(0,800):
        for b in range(0,1200):
            if otherblue[a][b] == True:
                blueother.append(im[a][b])
            else:
                other.append(im[a][b])

blist1 = np.array(blist)
backlist1 = np.array(backlist)
mean_blist = np.mean(blist1,axis = 0)
mean_backlist = np.mean(backlist1, axis = 0)
var_blist = np.var(blist1, axis = 0)
var_backlist = np.var(backlist1, axis = 0)
blueother1 = np.array(blueother)
mean_otherblue = np.mean(blueother1, axis = 0)
var_otherblue = np.var(blueother1, axis = 0)
