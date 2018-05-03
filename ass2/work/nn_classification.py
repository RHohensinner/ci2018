from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from nn_classification_plot import plot_hidden_layer_weights, plot_histogram_of_acc
from nn_classification_plot import plot_histogram_of_acc, plot_hidden_layer_weights, plot_random_images

from sklearn.model_selection import train_test_split
import numpy as np

__author__ = 'bellec,subramoney'

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""


def ex_2_1(input2, target2):
    """
    Solution for exercise 2.1
    :param input2: The input from dataset2
    :param target2: The target from dataset2
    :return:
    """
    
    #declaring variables used for MLPClassifier
    hidden_layers = 6
    solver_mode = 'adam'
    activation_mode = 'tanh'
    max_iter = 200

    cf = MLPClassifier(hidden_layer_sizes = (hidden_layers,), solver = solver_mode, activation = activation_mode, max_iter = max_iter)

    #training the classifier
    cf.fit(input2, target2[:,1])

    #calculate y_predicted and y_true for confusion matrix calculation

    #printing confusion matrix 
    print(confusion_matrix(target2[:,1], cf.predict(input2)))

    #plotting the hidden layer weights
    plot_hidden_layer_weights(cf.coefs_[0])

    pass


def ex_2_2(input1, target1, input2, target2):
    """
    Solution for exercise 2.2
    :param input1: The input from dataset1
    :param target1: The target from dataset1
    :param input2: The input from dataset2
    :param target2: The target from dataset2
    :return:
    """
    
    #declaring variables used for MLPClassifier
    hidden_layers = 20
    solver_mode = 'adam'
    activation_mode = 'tanh'
    max_iter = 1000

    max_accuracy = 0.0

    train_accuracy = []
    test_accuracy = []
    cfn = []
    
    m = 0

    for m in range(10):
        cf = MLPClassifier(hidden_layer_sizes = (hidden_layers,), activation = activation_mode, solver = solver_mode, 
        	               random_state = m, max_iter = max_iter)
        cf.fit(input1, target1[:,0])

        train_accuracy.append(cf.score(input1, target1[:,0]))

        current_test_accuracy = cf.score(input2, target2[:,0])

        test_accuracy.append(current_test_accuracy)

        plot_histogram_of_acc(train_accuracy[m], test_accuracy[m])

        if current_test_accuracy > max_accuracy:
        	cfn = confusion_matrix(target2[:,0], cf.predict(input2))
        	max_accuracy = current_test_accuracy

    print(cfn)
    
    #plot_histogram_of_acc(train_accuracy, test_accuracy)   
    #plot_random_images(input2)

    pass
