# Improved_object_detection
 In this project I will try to find a solution for this problem by using image processing techniques and algorithm for object detection.
 
In this project, I have used two different datasets from Kaggle and they are Dogs vs. Cats Redux: Kernels Edition and Chair-Kitchen-Knife-Saucepan. 


Different libraries and methods used are listed below.
a) Pandas: It is the library available in python which is
used to import and manage imported dataset to the program.
b) Matplotlib: It is the library of python that will be
used to show the output of the detection result for example
by plotting the result images.
c) Numpy: It is the library that is used to perform
mathematical calculations. Majorly used in machine
learning, data science, and any sort of program that deals
with high matrix computations.
d) Sklearn: This library of python is used for preprocessing the dataset and dividing it into testing and
training sets.
e) OpenCV: This is the most popular library of python
to deal with applications related to computer vision and
image processing.
f) Keras: It is high level API that is built over
Tensorflow and Theano. It is most easy to use library for
deep learning applications when compared to TensorFlow
(which is the most popular library used for deep learning).
g) Contrast Improvement: For contrast enhancement I
will be using CLAHE because it gives the best suited results
when compared to other methods for enhancement with
Gaussian Filtering.
h) Edge Enhancement: For edge enhancement I will
use Gaussian Filter because it smoothens the image while
sharpening the images so it is less prone to over sharpening
which degrades the accuracy of our model.
i) CNN (Convolution Neural Network): Convolution
Neural Network is essentially an Artificial Neural Network
(ANN) with large number of hidden layers. An ANN is a
machine learning technique modelled over the working
principle of the human brain.
B. Implementation
1) Algorithm: It is divided into three steps, first one is
contrast improvement, edge enhancement and object
detection.
a) Contrast Improvement: I will be using CLAHE
method which is relatively newer method when compared to
histogram equalization and in our case since we are
sharpening the image therefore when morphological
filtering method for contrast improvement is used it results
in a lot of white dots and appears to be not that effective.
b) Edge Enhancement: I will be using Gaussian filter
because it helps in smoothening of the image while
sharpening thus the image looks more natural.
Other methods like Laplacian can be seen to over sharpen
the image especially when they are combined with some
contrast improvement methods.
c) Object Classification: I will use CNN (convolution
neural network) for training and testing the model for object
detection part. It is essentially an ANN (artificial neural
network) with a lot of hidden layers. It has been designed to
resemble the working of human brains like the neurons,
dendrites, axion and synaptic gap we have artificial neuron,
net input, net output and weights associated with each input
or output. Like our brain has capability to learn things over
time by experience similarly the ANN/CNN has capability
to learn from multiple experiences/examples over a period
of time. It has a feed-forward structure meaning that the
neurons pass the information in single direction (from one
layer to another) and nothing is returned back to the
network. Its ability in object detection has been proven to be
quite accurate.
After running the classifier on the image without preprocessing the output of the classifier is found to be wrong.
But, when the image is pre-processed and the feeded to the
classifier for prediction the output is correct.
