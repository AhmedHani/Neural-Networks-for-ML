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


# Parameters
tf.flags.DEFINE_integer("epochs", 1000, "Training number of epochs")
tf.flags.DEFINE_integer("batch_size", 100, "Training batch size")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("Model: " + str("perceptron"))
print("Epochs: " + str(FLAGS.epochs))
print("Batch Size: " + str(FLAGS.batch_size))
#

cost_history = []
train_features, train_labels, test_features, test_labels = generate_linear_separable_data_for_binary_classifier(1000, 500)
weights = np.random.normal(size=len(train_features[0]))

for epoch in range(0, FLAGS.epochs):
    for i in range(0, len(train_features)):
        current_input = train_features[i]
        output = np.dot(weights.T, current_input)

        current_output = 1 if output > 0 else -1
        actual_output = train_labels[i]
        error = actual_output - current_output

        weights = np.add(weights, np.dot(current_input, error))

test_results = []

for i in range(0, len(test_features)):
    current_features = test_features[i]
    output = np.dot(weights.T, current_features)
    test_results.append(1 if output > 0 else -1)

print("Test accuracy: " + str(((np.count_nonzero(test_results == test_labels) * 1.0) / len(test_labels) * 1.0) * 100) + " %")