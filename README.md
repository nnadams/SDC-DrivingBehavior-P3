# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[center]: ./img/center.jpg "Center Camera"
[left]: ./img/left.jpg "Left Camera"
[right]: ./img/right.jpg "Right Camera"
[flipped]: ./img/flipped.jpg "Center Flipped"
[cropped]: ./img/cropped.jpg "Center Cropped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md (this file) which summarizes the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists a smaller version of the Nvidia self driving car network [https://devblogs.nvidia.com/deep-learning-self-driving-cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars). It is made up of many convolution layers with 5x5 and 3x3 filter sizes and depths between 24 and 64.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The "test set" for the model was running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

The training data was focused on center lane driving, smooth cornering and returning to the road center. I also drove the opposite way around the track to generalize the model.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to take full advantage of the feature extraction capabilities of convolutions layers.

My first step was to use a neural network structure similar to the I built for classifying traffic signs. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. This training and validation mean squared errors seemed a bit high, and driving with the model proved that it was not capable of making it around the track. Specifically sharp turns and the bridge (with bricks instead of pavement) caused the car to go off the track.

I moved on to Nvidia's model, linked above, but I removed some neurons from the fully connected layers (in the interest of training time mostly) and added a dropout layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle went off course. Notably still on the bridge and during sharp turns. I connected a Nintendo Switch controller to drive the simulator more smoothly and recollected data. This time I focused on cornering as smoothly as possible.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consists of the following layers, see Nvidia link above for their original model.

|      Layer      |              Description               |
| :-------------: | :------------------------------------: |
|      Input      |            320x160x3 image             |
|    Normalize    |           Mean normalization           |
|    Cropping     |     Remove top and bottom scenery      |
| Convolution 5x5 | 24 Filter, 2x2 Stride, RELU Activation |
| Convolution 5x5 | 36 Filter, 2x2 Stride, RELU Activation |
| Convolution 5x5 | 48 Filter, 2x2 Stride, RELU Activation |
| Convolution 3x3 |       64 Filter, RELU Activation       |
| Convolution 3x3 |       64 Filter, RELU Activation       |
| Fully Connected |              Outputs 512               |
|     Dropout     |          60% Training Dropout          |
| Fully Connected |              Outputs 256               |
| Fully Connected |               Outputs 64               |
|     Output      |             Steering input             |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving going counterclockwise. Here is example images of center lane driving from all three cameras:

![alt text][center]
![alt text][left]
![alt text][right]

Then I repeated this process on but clockwise in order to get more data points. Again the Keras cropping layer removes the highlighted portions of each image to avoid giving the model unnecessary complexities.

![alt text][cropped]

To augment the data sat, I also flipped images and angles thinking that this would help the model generalize. For example, here is an image that has then been flipped:

![alt text][center]
![alt text][flipped]

I then preprocessed this data by normalizing and cropping as described above. In total I had 50,688 data points  for training. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 or 3 as evidenced by the losses either stagnating or increasing if any more were done. I used an adam optimizer so that manually training the learning rate wasn't necessary.