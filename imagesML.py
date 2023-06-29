import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#----------------------Load the data from the dataset----------------------------------------------

# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist

# Load the training and test split of the Fashion MNIST dataset
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

#-------------------print out the training_images, training_labels----------------------------------

# You can put between 0 to 59999 here
index = 0

# Set number of characters per row when printing
np.set_printoptions(linewidth=320)

# Print the label and image
print(f'LABEL: {training_labels[index]}')
print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')

# Visualize the image
plt.imshow(training_images[index])

#-------------------print out a test_image, training_label--------------------------------------------

# You can put between 0 to 10000 here
index2= 0

# Set number of characters per row when printing
np.set_printoptions(linewidth=320)

# Print the label and image
print(f'LABEL: {test_labels[index2]}')
print(f'\nIMAGE PIXEL ARRAY:\n {test_images[index2]}')

# Visualize the image
plt.imshow(test_images[index2])

#-----------------------------------------------------------------------------------------------------
# Normalize the pixel values of the train and test images
training_images  = training_images / 255.0
test_images = test_images / 255.0

print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index2]}')
print(f'\nIMAGE PIXEL ARRAY:\n {test_images[index2]}')

#-----------------------------------------------------------------------------------------------------
# Build the classification model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Sequential: That defines a sequence of layers in the neural network.
# Flatten: Remember earlier where our images were a 28x28 pixel matrix when you printed them out? Flatten just takes that square and turns it into a 1-dimensional array.
# Dense: Adds a layer of neurons
# Each layer of neurons need an activation function to tell them what to do. There are a lot of options, but just use these for now:
# ReLU effectively means: 
# if x > 0: 
#   return x
# else: 
#   return 0
# In other words, it only passes values greater than 0 to the next layer in the network.
# Softmax takes a list of values and scales these so the sum of all elements will be equal to 1. When applied to model outputs, 
# you can think of the scaled values as the probability for that class. For example, in your classification model which has 10 units in the output dense layer,
# having the highest value at index = 4 means that the model is most confident that the input clothing image is a coat. If it is at index = 5, then it is a sandal,
# and so forth. See the short code block below which demonstrates these concepts. You can also watch this lecture if you want to know more about the Softmax function
# and how the values are computed.

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

# Evaluate the model on unseen data
model.evaluate(test_images, test_labels)

# # if you got your target percent on the training, it will stop the epoch
# class myCallback(tf.keras.callbacks.Callback):
#       def on_epoch_end(self, epoch, logs={}):
#     if(logs.get('accuracy') >= 0.6): # Experiment with changing this value
#       print("\nReached 60% accuracy so cancelling training!")
#       self.model.stop_training = True

# callbacks = myCallback()

# fmnist = tf.keras.datasets.fashion_mnist
# (training_images, training_labels) ,  (test_images, test_labels) = fmnist.load_data()

# training_images=training_images/255.0
# test_images=test_images/255.0
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(512, activation=tf.nn.relu),
#   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

