# Author: Ahmed Hani
# Package: https://github.com/AhmedHani/Neural-Networks-for-ML/tree/master/Implementations
#
# The package is implemented according to the lectures of Toronto University's Neural Networks for Machine Learning ..
# taught by Geoffrey Hinton.
#
# Course link: https://www.coursera.org/learn/neural-networks
# Lectures Repository: https://github.com/AhmedHani/Neural-Networks-for-ML/tree/master/Lectures
#
# Lecture link: https://www.coursera.org/learn/neural-networks/home/week/2

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_helpers import *


# Number of epochs that would be used for linear regression learning process (default: 1000)
tf.flags.DEFINE_integer("epochs", 200, "Training number of epochs")
tf.flags.DEFINE_float("learning_rate", 0.1, "Learning rate value for training phase")

# Parsing the arguments
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Model log data
print("Model: " + str("Online Linear Regression"))
print("Epochs: " + str(FLAGS.epochs))
print("Learning Rate" + str(FLAGS.learning_rate))

# Get training and testing data
train_features, train_labels, test_features, test_labels = get_data_for_stock_market_for_linear_regression()

# Initialize the linear regression weights in Gaussian Distribution
weights = np.random.normal(size=2)

# Training loop
for epoch in range(0, FLAGS.epochs):
    print("Epoch: " + str(epoch))

    for i in range(0, len(train_features)):
        # Get the current input from the training data
        current_input = train_features[i]

        # Calculate the output by multiplying the input vector with the weights
        current_output = np.dot(weights.T, current_input)[0]

        # Get the desired output of the data
        desired_output = train_labels[i]

        # Update the weights using the error according to the formula w(t+1) = w(t) - alpha * (1/m) * error
        weights[0] -= FLAGS.learning_rate * (1 / (len(train_labels) * 1.0)) * (desired_output - current_output) * current_input
        weights[1] -= FLAGS.learning_rate * (1 / (len(train_labels) * 1.0)) * (desired_output - current_output) * 1

# Testing loop
test_results = []
for i in range(0, len(test_features)):
    current_features = test_features[i]
    output = np.dot(weights.T, current_features)
    test_results.append(output)

plt.plot(test_features, test_labels, marker='o', color='r', ls='')
plt.plot([weights[0], -weights[0]], [-weights[1], weights[1]], marker='', color='b', ls='--')
plt.show()
