'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
from skimage.measure import label, regionprops
import numpy as np
from scipy.stats import multivariate_normal


class BarrelDetector():
    def __init__(self):
       '''
    	   Initilize your blue barrel detector with the attributes you need
   		   eg. parameters of your classifier
       '''
       
       self.mean_backlist = [88.6504, 100.006, 107.287]
       self.mean_blist = [124.575, 75.8289, 69.7989]
       self.mean_otherblue = [121.617, 97.4617, 75.3231]
       self.diag_backlist = [[3820.06, 0, 0], [0, 3567.51, 0], [0, 0, 3784.16]]
       self.diag_blist = [[3239.6, 0, 0], [0, 1732.05, 0], [0, 0, 1436.74]]
       self.diag_otherblue = [[2199.53, 0, 0], [0, 1469.77, 0], [0, 0, 1764.75]]

       
    def segment_image(self, img):
       '''
			Calculate the segmented image using a classifier
			eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		'''
       prior_back, prior_barrel = 0.75, 0.25
       prior_new, prior_otherblue = 0.6, 0.4
       
       norm_blist = multivariate_normal.pdf(img, self.mean_blist, self.diag_blist)
       norm_backlist = multivariate_normal.pdf(img, self.mean_backlist, self.diag_backlist)
       norm_otherblue = multivariate_normal.pdf(img, self.mean_otherblue, self.diag_otherblue)
       
       jbarrel = norm_blist * prior_barrel
       jback = norm_backlist * prior_back
       jbarrel1 = norm_blist * prior_new
       jother = norm_otherblue * prior_otherblue
       
       k1 = jbarrel > jback
       k1 = np.reshape(k1,(1,800*1200))
       k2 = jbarrel1 > jother
       k2 = np.reshape(k2,(1,800*1200))
       mask_img = np.zeros((1,800*1200))
       
       for i in range(len(k1[0])):
           if k1[0][i] == True and k2[0][i] == True:
               mask_img[0][i] = 1
               
       mask_img = np.reshape(mask_img,(800,1200))
       
       return mask_img     

    
    def get_bounding_box(self, img):
       '''
			Find the bounding box of the blue barrel
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
       mask_img = self.segment_image(img)
       result_after = mask_img.astype(np.uint8)
       kernel = np.ones((5,5), np.uint8)
       result_after = cv2.erode(result_after, kernel)
       result_after = cv2.dilate(result_after,kernel,iterations = 3)
       contours, hierarchy = cv2.findContours(result_after, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       boxes = []
       for i in range(len(contours)):
           cnt = contours[i]
           x,y,w,h = cv2.boundingRect(cnt)
           if h/w < 1.4 or h/w > 3:
               continue
           else:
               rect = cv2.minAreaRect(cnt)
               box = np.int0(cv2.boxPoints(rect))
               box = box.tolist()
               boxes.append([x,y,x+w,y+h])   

       boxes = sorted(boxes, key=lambda x: x[1])
            
       return boxes


#if __name__ == '__main__':
#	folder = "trainset"
#	my_detector = BarrelDetector()
#	for filename in os.listdir(folder):
#		# read one test image
#		img = cv2.imread(os.path.join(folder,filename))
#		cv2.imshow('image', img)
#		cv2.waitKey(0)
#		cv2.destroyAllWindows()

		#Display results:
		#(1) Segmented images
		#	 mask_img = my_detector.segment_image(img)
		#(2) Barrel bounding box
		#    boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope

