# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./traffic.png "Visualization"
[image2]:./bar_traffic.png "Bar Visualization"
[image3]: ./traffic_sign_origninal.png "Original"
[image3.5]: ./traffic_sign_gray.png "Grayscaling"
[image4]: ./test_images/11.jpg "Traffic Sign 1"
[image5]: ./test_images/12.jpg "Traffic Sign 2"
[image6]: ./test_images/13.jpg "Traffic Sign 3"
[image7]: ./test_images/14.jpg "Traffic Sign 4"
[image8]: ./test_images/15.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the numbers of each of the classes.

![alt text][image2]

Another exploratory visualization of the data set is a chart showing the representative images of each of the classes.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because this will increase contrast and save computational resources.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]
![alt test][image3.5]

As a last step, I normalized the image data so that all the images are uniform: they have zero mean and equal variance.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5		| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5		| 1x1 stride, same padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 5x5x16   				|
| Flatten   			| outputs 400									|
| Fully connected		| outputs 120									|
| RELU					|												|
| Fully connected		| outputs 83									| 
| RELU					|												|
| Fully connected		| outputs 43									| 
| Softmax				|												|
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer. The batch size is 128. The number of epochs is 100 and the learning rate is 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.947
* test set accuracy of 0.929

I first chose the LeNet architechture because it is an excellent starting point for CNN training. It is not too large or too small. It implements multiple {convolution + max-pooling} layers. I started off with a learning rate of 0.001. I tried to add dropout layers between fully connected layers but the accuracy seems to become worse. I think it's because the network is not very deep, so regularization doesn't seem to be necessary. Changing the learning rate didn't help. Increasing the learning rate seems to improve the validation accuracy, but significantly decrease the test accuracy, suggesting overfitting. I tried different batch sizes. Increasing the batch to 256 or decreasing it to 64 doesn't seem to help. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because the triangular shape is similar to several other signs in the training set and the inside black shape is similar to that of the Pedestrians sign. The second image might be difficult to classify because the white quadrangle on the edge is similar to the white circular arrows of the Roundabout mandatory sign. The third image might be difficult to classify because the triangular shape is similar to several other signs in the training set. The Stop sign might be confused with the No entry sign because both signs have a pretty big red area and white area inside. The No vehicles sign might be confused with the speed limit signs because they all have a big red circle.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image								   |     Prediction	        						| 
|:------------------------------------:|:----------------------------------------------:| 
| Right-of-way at the next intersection| Right-of-way at the next intersection			| 
| Priority road						   | Priority road									|
| Yield								   | Yield											|
| Stop								   | Stop											|
| No vehicles						   | No vehicles									|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.929. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 22th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Right-of-way at the next intersection sign (probability of 1.00000000e+00), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00 		| Right-of-way at the next intersection			| 
| 3.18842655e-36		| Pedestrians									|
| 0.00000000e+00		| Speed limit (20km/h)							|
| 0.00000000e+00		| Speed limit (30km/h)			 				|
| 0.00000000e+00	    | Speed limit (50km/h)							|


For the second image, the model is relatively sure that this is a Priority road sign (probability of 9.92897511e-01), and the image does contain a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.92897511e-01 		| Priority road									| 
| 7.10250577e-03		| Roundabout mandatory							|
| 4.96911594e-13		| Right-of-way at the next intersection			|
| 1.05051508e-14		| Turn right ahead			 					|
| 8.75557364e-16	    | End of no passing by vehicles over 3.5 metric tons|

For the third image, the model is relatively sure that this is a Yield sign (probability of 1.), and the image does contain a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1. 					| Yield											| 
| 0.					| Speed limit (20km/h)							|
| 0.					| Speed limit (30km/h)							|
| 0.					| Speed limit (50km/h)			 				|
| 0.					| Speed limit (60km/h)							|

For the fourth image, the model is relatively sure that this is a Stop sign (probability of 1.00000000e+00), and the image does contain a Stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00		| Stop											| 
| 2.23172481e-18		| No entry										|
| 9.37442918e-26		| Yield											|
| 1.83648687e-27		| Turn left ahead			 					|
| 6.90329727e-30		| Roundabout mandatory							|

For the fifth image, the model is relatively sure that this is a No vehicles sign (probability of 1.00000000e+00), and the image does contain a No vehicles sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00		| No vehicles									| 
| 4.21008445e-34		| Speed limit (100km/h)							|
| 3.35491856e-35		| Speed limit (50km/h)							|
| 3.22660141e-36		| Speed limit (80km/h)		 					|
| 0.00000000e+00		| Speed limit (20km/h)							|




