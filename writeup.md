**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/TrainingDataSample.png
[image2]: ./output_images/HogFeaturesExtraction.png
[image3]: ./output_images/CarDetectionProcess.png
[video1]: ./result.mp4
[video2]: ./result-extra.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step appears in the code cells 2 and 3 of the IPython notebook [P5.ipynb](./P5.ipynb), in here the methods to extract color features and HOG features are defined, they are used later to prepare the classifier for training (code cell 7) given the training and test set, and to do the identifications (code cell 10) of cars in the video's images.

First, I extracted the `vehicle` and `non-vehicle` images, I tried with different set sizes, a size of 10,000 (5,000 for each type) produced result with more accuracy during the trainning.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Then, I was trying different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

After several attemps with multiple combinations, I noticed than using the color space `YCrCb` was getting more accurate results during the training, HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` seemed to produce good results with a balance between accuracy and training/process speed.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

During my tests with different training set sizes and models, I realized that using color features (spatially binned color and histograms of color) in addition to HOG features in the 3 channels of YCrCb color space, resulted in an increment on the accuracy. So my final decition was to use a combination of them, this can be seen in the cell code 3 of the IPython notebook [P5.ipynb](./P5.ipynb), where both kind of features extraction is performed in the method `extract_features`. As recomended during the lessons the HOG features are extracted for the entire image to improve the performance.

The training is implemeted in the code cells 7 and 8 of the IPython notebook [P5.ipynb](./P5.ipynb). I did several attemps using diferent types of classifiers, I used Linear SVM, SVM (with different core sizes) and Decision Trees. At the end, I decided to use a linear SVM which was the one presented the better balance between accuracy and speed. In genetal SVC could achieve higher accuracy levels but the model was very costly during the identification process and had tedency to over fit. Decition Trees were very fast for training and identification, however the level of achievable accuracy was not satisfactory in practice (just above 0.8).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to use a window sliding mechanism which is implemented in code cell 10 of the IPython notebook [P5.ipynb](./P5.ipynb), it is restricted to the lower part of the image (where the cars can be present). I chose to do the sliding windows process several times for scales 1.0, 1.5, 2.0 and 3.0 in order to do the identification for different sizes and create enoght boxes to later do a significat contribution in the heat map (mechanism used to mitigate false positives).

The following image is an example of the mechanism used to do the identification, the first image from left to right shows the boxes retrieved by the sliding window search applied several times according to the mentiones scales.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

At the end, the search takes place on the scaled images using the HOG features for the 3 YCrCb channels plus spatially binned color and histograms of color (in the same color space) in the feature's vector, which provided a satisfactory result

As stated before, I was using a sliding window search applied several times for the scales 1.0, 1.5, 2.0 and 3.0. The mitigation of overlaping and false positives was done using a heat map with a threshold, the implementation can be seen also in code cell 10 of the IPython notebook [P5.ipynb](./P5.ipynb).

An additional mechanism to mitigate false positives was to introduce the concept of heat dilution (I used a rate of 0.5), that means that the detections of previos images also contribute adding heat to the heat map, but not with the same intensity as the current one, and that contribution is reduced as long as the image gets older.

This (previously presented) image is an example of the mechanism used to do the identification, the second image from left to right shows the heat map resulting after the identification of boxes by the sliding window search, and the third image shows the identification resulted after apliying the threshold to the heat map.

![alt text][image3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

This is the link to my resultig video [result.mp4](./result.mp4).

And this link is to a video that combines the vehicle detection with the lane deection (implemented the previous project) [result-extra.mp4](./result-extra.mp4), the implementaion of this combination can be found in the the IPython notebook [P5-extra.ipynb](./P5-extra.ipynb).



#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As stated before, I used a heatmap and then thresholded to identify vehicle positions, in order to do that I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap, assuming than each blob corresponded to a vehicle,  from the blobs the bounding boxes are constructed to cover the area that each blob detected.

Same image as before shows how this mechanim which takes place:

![alt text][image3]

Also as mentioned before, heat dilution (with a rate of 0.5) is used in order to make the previous images to contribute adding heat, but their heat diluting as they are getting older. The intuition behind that is that for actual cars its heat is not diluted (maybe moving across the images), but for false positives their heat quickly disapears.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem I faced was the time that takes to do the identification process for each frame of the video (even when doing the HOG features extraction for the entire image saves some time), the more complex the classifier's model is the more time it takes, for this reason a Linear SVC was chosen which seems to present and acceptable balance between accuracy and performance.

Even with a relatively somple model, the procesing time to do the identification seem to be hardly suitable to be used in real time at least for an average computer. A solution I came accross was to not perform the identification in every frame, but every certain numbers of frames instead. The intuition behind that is that the vehicles are not going to disapear sudenly or move to extraorinary speeds, so even doing the identification a few times per second would be enough for the detection mechanism to work properly, in the heat map the vehicules would generate enough heat to track the down.

Also as the optional challenge, I combined the cars detection of this project with the lane detection of the previous one, the implementation can be found in the IPython notebook [P5-extra.ipynb](./P5-extra.ipynb) and the result in the video [result-extra.mp4](./result-extra.mp4).

