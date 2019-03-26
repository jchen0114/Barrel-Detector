# Color Segmentation
----
Classify images using single Gaussian, apply a bounding box onto the target item on image.

## Requirements
----
 - opencv-python>=3.4
 - matplotlib>=2.2
 - numpy>=1.14
 - scikit-image>=0.14.0

## Usage
----
#### Start Label

```sh
label.py
```
This part helps us hand label our training set by clicking on images and bound them. From this file, simply import the image your disire to label. After starting the code, click left to start bounding box and clikc right to stop funciton. It will automatically save as an npy file after label. We label each image twice, the first time we label the barrels, the second time we label the items that are blue but not barrels.
#### Get Parameters
```sh
parameters.py
```
This file imports the image we labeled then masked the corrsponding location onto an actual image. After seperating the barrels from background, we extract the RGB value from each pixel and stack them into an array. Then we calculate the mean and covaraince for each part. The code actually ran three sets of mean and variance, the barrels, background, and non-barrel blue items.
#### Barrel Detector
```sh
barrel_detector.py
```
This code is the code we run for our results. There are three def in the class function, init, segment_image and get_bounding_box. 
For init, we store our parameters from parameter.py to avoid running unnecessary process everytime. 
In segment_image, we ran three single Gaussian function to calcuate the maximum likelihood, then multiply them by the prior to apply Bayes decision rule.
Last in get_bounding_box, we use the results from segment_image at inputs to plot the bounding boxes on the original image. We set several boundaries to filter out bounding boxes that do not fit the barrel size or shape. We output the coordinate of the top left corner and width and length for each bounding box([x,y,x+w,y+h]).



### Reference
----
 -  hand-labeling: roipoly: https://github.com/jdoepfert/roipoly.py
 - conversion: cvtColor: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html 



