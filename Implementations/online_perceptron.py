# Author: Ahmed Hani
# Package: https://github.com/AhmedHani/Neural-Networks-for-ML/tree/master/Implementations
#
# The package is implemented according to the lectures of Toronto University's Neural Networks for Machine Learning ..
# taught by Geoffrey Hinton.
#
# Course link: https://www.coursera.org/learn/neural-networks
# Lectures Repository: https://github.com/AhmedHani/Neural-Networks-for-ML/tree/master/Lectures
#
# Lecture link: https://www.coursera.org/learn/neural-networks/home/week/1


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_helpers import generate_linear_separable_data_for_binary_classifier


# Number of epochs that would be used for perceptron learning process (default: 1000)
tf.flags.DEFINE_integer("epochs", 1000, "Training number of epochs")

# Parsing the arguments
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Model log data
print("Model: " + str("perceptron"))
print("Epochs: " + str(FLAGS.epochs))
print("Batch Size: " + str(FLAGS.batch_size))

# Get training and testing data
train_features, train_labels, test_features, test_labels = generate_linear_separable_data_for_binary_classifier(1000, 500)

# Initialize the perceptron weights in Gaussian Distribution
weights = np.random.normal(size=len(train_features[0]))

# Training loop
for epoch in range(0, FLAGS.epochs):
    print("Epoch: " + str(epoch))

    for i in range(0, len(train_features)):
        # Get the current input from the training data
        current_input = train_features[i]

        # Calculate the output by multiplying the input vector with the weights
        output = np.dot(weights.T, current_input)

        # Set the output to 1 if the output value is higher than 0 and to -1 if less
        current_output = 1 if output > 0 else -1

        # Get the desired output of the data
        desired_output = train_labels[i]

        # Calculate the error by subtracting the desired output by the actual output
        error = desired_output - current_output

        # Update the weights using the error according to the formula w(t+1) = w(t) + ((d - y) * x)
        weights = np.add(weights, np.dot(current_input, error))

# Testing loop
test_results = []
for i in range(0, len(test_features)):
    current_features = test_features[i]
    output = np.dot(weights.T, current_features)
    test_results.append(1 if output > 0 else -1)

print("Test accuracy: " + str(((np.count_nonzero(test_results == test_labels) * 1.0) / len(test_labels) * 1.0) * 100) + " %")

plt.plot(test_features[:, 0], test_features[:, 1], marker='o', color='r', ls='')
plt.plot([weights[0], -weights[0]], [-weights[1], weights[1]], marker='', color='b', ls='--')
plt.show()
