# Author: Ahmed Hani
# Package: https://github.com/AhmedHani/Neural-Networks-for-ML/tree/master/Implementations
#
# The package is implemented according to the lectures of Toronto University's Neural Networks for Machine Learning ..
# taught by Geoffrey Hinton.
#
# Course link: https://www.coursera.org/learn/neural-networks
# Lectures Repository: https://github.com/AhmedHani/Neural-Networks-for-ML/tree/master/Lectures

import numpy as np


def generate_linear_separable_data_for_binary_classifier(train_size, test_size):
    """
    Generate 2D random linear-separable features for linear binary classifiers
    :param train_size: An integer that specifies and the number of training data
    :param test_size: An integer that specifies and the number of testing data
    :return: lists of train features, train labels, test features and test labels
    """

    train_features, train_labels, test_features, test_labels = [], [], [], []

    for i in range(1, train_size + 1):
        if i % 2:
            train_features.append([((np.random.uniform() * 2 - 1) / 2 - 0.5), ((np.random.random() * 2 - 1) / 2 + 0.5)])
            train_labels.append(1)
        else:
            train_features.append([((np.random.uniform() * 2 - 1) / 2 + 0.5), ((np.random.random() * 2 - 1) / 2 - 0.5)])
            train_labels.append(-1)

    for i in range(1, test_size + 1):
        if i % 2:
            test_features.append([((np.random.uniform() * 2 - 1) / 2 - 0.5), ((np.random.random() * 2 - 1) / 2 + 0.5)])
            test_labels.append(1)
        else:
            test_features.append([((np.random.uniform() * 2 - 1) / 2 + 0.5), ((np.random.random() * 2 - 1) / 2 - 0.5)])
            test_labels.append(-1)

    return np.array(train_features), np.array(train_labels), np.array(test_features), np.array(test_labels)




