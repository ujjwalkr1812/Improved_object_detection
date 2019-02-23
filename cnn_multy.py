# ---*-----*---Part 1 - Building the CNN---*-------*---

# Importing the Keras libraries and packages
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# parameter1 - number of filters to be used
# parameter2 - dimensions of the filters used
# parameter3 - dimensions of image like length, breadth and 3 (for color images)
# parameter4 - activation function to be used to bring non-linearity
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
# parameters represent the dimensions of the pool filter
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional and pooling layer for better accuracy
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening the convolved and pooled input matrices creating the input for the ConvNet
classifier.add(Flatten())

# Step 4 - Full connection
# Hidden Layer
classifier.add(Dense(units = 128, activation = 'relu'))
# Output Layer
classifier.add(Dense(units = 5, activation = 'softmax'))

# Compiling the CNN
# optimizer is used to 
# loss and metrics are similar and are used to judge the performance of your model
# But only loss function results are used when training the model
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



# ---*---*----Part 2 - Fitting the CNN to the images---*------*---

# Importing the libraries and packages
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile

# To prevent model from stopping while running if any currupted image is read from dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Image augmentation which is basically pre-processing of images
# This generates batches of image data and is important to prevent overfitting
train_datagen = ImageDataGenerator(
        rescale = 1./255, # Rescales the value of pixels between 0 and 1 (feature-scaling)
        shear_range = 0.2, # max random shear range (its a geometric transformation)
        zoom_range = 0.2, # min(1-zoom_range) and max(1+zoom_range) random zoom alowed
        horizontal_flip = True # Random horizontal flipping of input is allowed
        )
# Image augmentation of test data (in this case only feature-scaling has been done) 
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creates the training set according to the image augmentation - "train_datagen"
training_set = train_datagen.flow_from_directory(
        'Dataset_Multy/training_set', # Dataset source directory
        target_size = (64, 64), # size of data as expected in CNN model (input_shape)
        batch_size = 32, # no. of images after which the weights will be updated in CNN
        class_mode = 'categorical' # indicates no. of classes is more than 2
        )

# Creates the testing set according to the image augmentation - "test_datagen"
test_set = test_datagen.flow_from_directory(
        'Dataset_Multy/testing_set', # Dataset source directory
        target_size = (64, 64), # size of data as expected in CNN model (input_shape)
        batch_size = 32, # no. of images after which the weights will be updated in CNN
        class_mode = 'categorical' # indicates no. of classes is more than 2
        )

# This the main code that is used to fit CNN on the training set
# and test the performance of it on the testing set in each epoch
classifier.fit_generator(
        training_set, # It is the training set
        steps_per_epoch = 5915, # total no. of images in the training set
        epochs = 5, # no. of epochs to be conducted while training
        validation_data = test_set, # It is the test set on which we want to validate our model
        validation_steps = 1557 # no. of images in the testing set
        )


# ---*------*---Part 3 - Making new predictions---*------*---

# Importing the libraries and packages
import numpy as np
from tensorflow.python.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the image on which we need to make the predictions in 2D-array
test_image = image.load_img('Dataset_Multy/single_prediction/knife-darkfinal2Color.jpeg', target_size = (64, 64))

# Since input shape was 3D-array (colored image) so we need to convert test image to 3D-array
test_image = image.img_to_array(test_image)

# Add a new dimension to the test_image (4D-array) which corresponds to the "batch"
# Reason - predict method cannot accept a single input, i.e. inputs must be in form of batch
# Note- axis is position where new dimension is to be added and "predict" method expects it to be 0
test_image = np.expand_dims(test_image, axis = 0)

# Result of prediction
result = classifier.predict(test_image)

# The class indices which tell us mapping between class and their associated neumeric values
#training_set.class_indices


# Store the prediction result and print
prediction = "cat-"+str(result[0][0])
prediction += "\n chair-"+str(result[0][1])
prediction += "\n dog-"+str(result[0][2])
prediction += "\n kitchen-"+str(result[0][3])
prediction += "\n knife-"+str(result[0][4])
print("Prediction - \n",prediction)

# Input image
plt.figure(1)
plt.title('Knife')
plt.axis('off')
plt.imshow(image.load_img('Dataset_Multy/single_prediction/knife-darkfinal2Color.jpeg'))
#knife-dark.jpeg
#knife-darkfinal2Color.jpeg
#chair2.jpeg
#chair2final2Color.jpeg

# ---*-----*---Part 4 - Saving the trained model and weights---*------*----

# serialize model to JSON
model_json = classifier.to_json()
with open("model_cnn_multy.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to hdf5
classifier.save_weights("weights_cnn_multy.h5")



# ---*-----*---Part 5 - Loading the trained model weights---*------*----

classifier.load_weights("weights_cnn_multy.h5")



