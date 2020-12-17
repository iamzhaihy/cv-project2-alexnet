# CV Project 2: AlexNet

Hongyu Zhai (`hz2162`)



## Changes I Made

The project proposal was sent to Professor Wong on Nov 22. After that, I realized the download link to the training dataset ([LSVRC 2010](http://image-net.org/challenges/LSVRC/2010/download-public)) no longer works. With Professor's permission, I decided to use the [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

As the result of switching dataset, I made some changes to the AlexNet model as well. The original model expects the input shape to be $224 \times 224 \times 3$, but images in CIFAR-100 have much lower resolution $32 \times 32 \times 3$. Also, there are $10000$ unique class labels in the LSVRC 2010 dataset, but there are only $100$ in CIFAR-100. To deal with this difference in size and number, I changed the `input_shape` and the size of the output layer. More details can be found in `Implementing AlexNet.ipynb`.



## To Run the Program

This project requires the following dependencies to work

- `sklearn`: for assessing the performance
- `matplotlib`: for displaying images and making plots
- `tensorflow.keras`: for constructing the neural network 
  - also for loading the CIFAR-100 dataset

Two versions are included: `Implementing AlexNet.py`, and `Implementing AlexNet.ipynb`. The Jupyter Notebook version is recommended, because the user can explore the code step by step.



## The Source Code

```python
# # CS 6433 Project 2: Implementing AlexNet
# 
# Hongyu Zhai (`hz2162`)

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Loading the Dataset
# Keras provides a function to load CIFAR-100
from tensorflow.keras.datasets import cifar100
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode="fine")

n_test = X_test.shape[0]
n_train = X_train.shape[0]
img_shape = X_train.shape[1:]

print("Number of training examples:", n_train)
print("Number of testing examples:", n_test)
print("Shape of input images:", img_shape)

# First Convolution Layer
# 
# > "The first convolutional layer filters the 224×224×3 input image with 96 kernels of size 11×11×3 with a stride of 4 pixels."
# > "The ReLU non-linearity is applied to the output of every convolutional and fully-connected layer."
# > "Response-normalization layers follow the first and second convolutional layers."
# > "Max-pooling layers, of the kind described in Section 3.4, follow both response-normalization layers as well as the fifth convolutional layer."

# layer #1 conv #1
l1_conv1 = keras.layers.Conv2D(input_shape=img_shape,
                               filters=96,
                               kernel_size=(11, 11),
                               strides=(4, 4),
                               padding='same',
                               activation='relu')

# response-normalization layer follows the first conv layer
l1_conv1_norm = keras.layers.BatchNormalization()

# max pooling layer with s = 2, and z = 3
l1_conv1_pool = keras.layers.MaxPooling2D(pool_size=(3, 3),
                                          strides=(2,2),
                                          padding='same')

# Second Convolution Layer
# 
# > "The second convolutional layer takes as input the (response-normalized and pooled) output of the first convolutional layer and filters it with 256 kernels of size 5 × 5 × 48"

# layer #2 conv #2
l2_conv2 = keras.layers.Conv2D(filters=256,
                               kernel_size=(5, 5),
                               strides=(1, 1), 
                               padding='same',
                               activation='relu')

# response-normalization layer follows the second conv layer
l2_conv2_norm = keras.layers.BatchNormalization()

# max pooling layer with s = 2, and z = 3
l2_conv2_pool = keras.layers.MaxPooling2D(pool_size=(3, 3),
                                          strides=(2, 2),
                                          padding='same')

# Third Convolution Layer
# 
# > "The third, fourth, and fifth convolutional layers are connected to one another without any intervening pooling or normalization layers."
# > "The third convolutional layer has 384 kernels of size 3 × 3 × 256 connected to the (normalized, pooled) outputs of the second convolutional layer."

# layer #3 conv #3
l3_conv3 = keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=(1, 1), 
                               padding='same',
                               activation='relu')

# Fourth Convolution Layer
# 
# > "The fourth convolutional layer has 384 kernels of size 3 × 3 × 192"

# layer #4 conv #4
l4_conv4 = keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=(1, 1), 
                               padding='same',
                               activation='relu')

# Fifth Convolution Layer
# 
# > "the fifth convolutional layer has 256 kernels of size 3 × 3 × 192."

# layer #5 conv #5
l5_conv5 = keras.layers.Conv2D(filters=256,
                               kernel_size=(3, 3),
                               strides=(1, 1), 
                               padding='same',
                               activation='relu')

# max pooling layer with s = 2, and z = 3
l5_conv5_pool = keras.layers.MaxPooling2D(pool_size=(3, 3),
                                          strides=(2,2),
                                          padding='same')

#  First Fully-Connected Layer
# 
# > "The fully-connected layers have 4096 neurons each."
# > "We use dropout in the first two fully-connected layers."

# flatten before feeding to FC layers
l6_fc1_flat = keras.layers.Flatten()

# layer #6 fc #1
l6_fc1 = keras.layers.Dense(4096,
                            input_shape=(32,32,3,),
                            activation='relu')

# dropout with rate 0.5
l6_fc1_dropout = keras.layers.Dropout(0.5)

# Second Fully-Connected Layer

# layer #7 fc #2
l7_fc2 = keras.layers.Dense(4096,
                            activation='relu')

# dropout with rate 0.5
l7_fc2_dropout = keras.layers.Dropout(0.5)

# Third Fully-Connected Layer
# 
# > "The output of the last fully-connected layer is fed to a 1000-way softmax which produces a distribution over the 1000 class labels"

# layer #8 fc #3
l8_fc3 = keras.layers.Dense(100,
                            activation='softmax')

# Put Everything Together

AlexNet = keras.models.Sequential()

# first conv layer
AlexNet.add(l1_conv1)
AlexNet.add(l1_conv1_norm)
AlexNet.add(l1_conv1_pool)

# second conv layer
AlexNet.add(l2_conv2)
AlexNet.add(l2_conv2_norm)
AlexNet.add(l2_conv2_pool)

# third conv layer
AlexNet.add(l3_conv3)

# fourth conv layer
AlexNet.add(l4_conv4)

# fifth conv layer
AlexNet.add(l5_conv5)
AlexNet.add(l5_conv5_pool)

# first fc layer
AlexNet.add(l6_fc1_flat)
AlexNet.add(l6_fc1)
AlexNet.add(l6_fc1_dropout)

# second fc layer
AlexNet.add(l7_fc2)
AlexNet.add(l7_fc2_dropout)

# third fc layer
AlexNet.add(l8_fc3)

# compile the Sequential model 
AlexNet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Testing Our Model

# one-hot encoding the labels
from tensorflow.keras.utils import to_categorical

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

# train the model using training images
AlexNet.fit(X_train, y_train, batch_size=32, epochs=10)

# make predictions on testing images
y_predicted = AlexNet.predict(X_test)

# report accuracy score of the model
from sklearn.metrics import accuracy_score

score = accuracy_score(np.argmax(y_predicted, axis=1),
                       np.argmax(y_test, axis=1))

print("The accuracy score of our model is", score)
```







