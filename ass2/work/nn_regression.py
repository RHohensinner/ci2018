import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import matplotlib.pyplot as plt

from nn_regression_plot import plot_mse_vs_neurons, plot_mse_vs_iterations, plot_learned_function, \
    plot_mse_vs_alpha, plot_bars_early_stopping_mse_comparison

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""

__author__ = 'bellec,subramoney'


def calculate_mse(nn, x, y):
    """
    Calculate the mean squared error on the training and test data given the NN model used.
    :param nn: An instance of MLPRegressor or MLPClassifier that has already been trained using fit
    :param x: The data
    :param y: The targets
    :return: Training MSE, Testing MSE
    """
    ## TODO

    y_true = y
    y_pred = nn.predict(x)

    mse = mean_squared_error(y_true, y_pred)
    return mse


def ex_1_1_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    # declaring hidden layer neurons 2, 8, 40
    hidden_layer_2  = 2
    hidden_layer_8  = 8
    hidden_layer_40 = 40

    # declaring variables used in MLP-Regressor
    activation_mode = 'logistic'
    solver_mode = 'lbfgs'
    alpha = 0
    max_iter = 200



    # declaring MLP-Regressor:
    nn_2 = MLPRegressor(hidden_layer_sizes = (hidden_layer_2,), activation = activation_mode, 
    					solver = solver_mode, alpha = alpha, max_iter = max_iter)
    nn_8 = MLPRegressor(hidden_layer_sizes = (hidden_layer_8,), activation = activation_mode, 
    					solver = solver_mode, alpha = alpha, max_iter = max_iter)
    nn_40 = MLPRegressor(hidden_layer_sizes = (hidden_layer_40,), activation = activation_mode, 
    					solver = solver_mode, alpha = alpha, max_iter = max_iter)

    # train neural network using the regressor method fit
    nn_2.fit(x_train, y_train)
    nn_8.fit(x_train, y_train)
    nn_40.fit(x_train, y_train)

    # compute the output using the method predict
    y_test_pred_2 = nn_2.predict(x_test)
    y_train_pred_2 = nn_2.predict(x_train)
    y_test_pred_8 = nn_8.predict(x_test)
    y_train_pred_8 = nn_8.predict(x_train)
    y_test_pred_40 = nn_40.predict(x_test)
    y_train_pred_40 = nn_40.predict(x_train)

    # plotting learned function
    plot_learned_function(n_hidden = hidden_layer_2, x_train = x_train, y_train = y_train,
    					  y_pred_train = y_train_pred_2, y_pred_test = y_test_pred_2, x_test = x_test, y_test = y_test)
    plot_learned_function(n_hidden = hidden_layer_8, x_train = x_train, y_train = y_train,
    					  y_pred_train = y_train_pred_8, y_pred_test = y_test_pred_8, x_test = x_test, y_test = y_test)
    plot_learned_function(n_hidden = hidden_layer_40, x_train = x_train, y_train = y_train,
    					  y_pred_train = y_train_pred_40, y_pred_test = y_test_pred_40, x_test = x_test, y_test = y_test)

 	
    pass


def ex_1_1_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    
    m = 0
    # declaring variables used in MLP-Regressor
    hidden_layer  = 8
    random_state = 10
    activation_mode = 'logistic'
    solver_mode = 'lbfgs'
    alpha = 0
    max_iter = 200

    # declaring to arrays used to store train/test values
    train_mse = np.zeros((10, 1))
    test_mse = np.zeros((10, 1))

    for m in range(random_state):
    	nn = MLPRegressor(hidden_layer_sizes = (hidden_layer,), activation = activation_mode, 
    					  solver = solver_mode, alpha = alpha, max_iter = max_iter, random_state = m)
    	nn.fit(x_train, y_train)
    	#calculate for every random seed train_mse and test_mse
    	train_mse[i] = calculate_mse(nn, x_train, y_train)
    	test_mse[i] = calculate_mse(nn, x_test, y_test)

    #calculate mse test/train min/max/mean/stdd using argmin/argmax/mean/std
    min_test_mse = min(test_mse)
    min_train_mse = min(train_mse)

    max_test_mse = max(test_mse)
    max_train_mse = max(train_mse)

    mean_test_mse = np.mean(test_mse)
    mean_train_mse = np.mean(train_mse)

    stdd_test_mse = np.std(test_mse)
    stdd_train_mse = np.std(train_mse)    

    print("TEST:\n")
    print("Min: ", min_test_mse, "at ", np.argmin(test_mse), "\n")
    print("Max: ", max_test_mse, "at ", np.argmax(test_mse), "\n")
    print("Mean: ", mean_test_mse, "\n")
    print("Standard Deviation: ", stdd_test_mse, "\n")

    print("TRAIN:\n")
    print("Min: ", min_train_mse, "at ", np.argmin(train_mse), "\n")
    print("Max: ", max_train_mse, "at ", np.argmax(train_mse), "\n")
    print("Mean: ", mean_train_mse, "\n")
    print("Standard Deviation: ", stdd_train_mse, "\n")

    pass


def ex_1_1_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 c)
    Remember to set alpha to 0 when initializing the model
    Use max_iter = 10000 and tol=1e-8
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    m = 0
    n = 0
    # declaring variables used in MLP-Regressor
    hidden_layers  = np.array([1, 2, 3, 4, 6, 8, 12, 20, 40])
    random_state = 10

    activation_mode = 'logistic'
    solver_mode = 'lbfgs'
    alpha = 0
    max_iter = 10000
    tol = 1e-8

    train_mse = np.zeros((hidden_layers.size, random_state))
    test_mse = np.zeros((hidden_layers.size, random_state))

    for m in range(random_state):
        for n in range(hidden_layers.size):
            nn = MLPRegressor(hidden_layer_sizes = (hidden_layers[n],), activation = activation_mode, solver = solver_mode, 
                              alpha = alpha, max_iter = max_iter, random_state = m, tol = tol)
            nn.fit(x_train, y_train)
            #calculate for every random seed train_mse and test_mse
            train_mse[n][m] = calculate_mse(nn, x_train, y_train)
            test_mse[n][m] = calculate_mse(nn, x_test, y_test)

    plot_mse_vs_neurons(train_mse, test_mse, hidden_layers)

    y_test_pred = nn.predict(x_test)
    y_train_pred = nn.predict(x_train)

    plot_learned_function(n_hidden = 40, x_train = x_train, y_train = y_train,
    					  y_pred_train = y_train_pred, y_pred_test = y_test_pred, x_test = x_test, y_test = y_test)

    pass


def ex_1_1_d(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    Use n_iterations = 10000 and tol=1e-8
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    m = 0
    n = 0
    # declaring variables used in MLP-Regressor
    hidden_layers  = np.array([2, 8, 40])
    random_state = 10

    activation_mode = 'logistic'
    solver_mode = 'lbfgs'
    alpha = 0
    max_iter = 10000
    tol = 1e-8

    train_mse = np.zeros((hidden_layers.size, max_iter))
    test_mse = np.zeros((hidden_layers.size, max_iter))

    for m in range(3):
        nn = MLPRegressor(hidden_layer_sizes = (hidden_layers[m],), activation = activation_mode, solver = solver_mode, 
                              alpha = alpha, max_iter = 1, random_state = m, tol = 1e-8)
        for n in range(max_iter):
            nn.fit(x_train, y_train)
            # calculate for every random seed train_mse and test_mse
            train_mse[m][n] = calculate_mse(nn, x_train, y_train)
            test_mse[m][n] = calculate_mse(nn, x_test, y_test)

    plot_mse_vs_iterations(train_mse, test_mse, max_iter, hidden_layers)

    pass


def ex_1_2_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    ## TODO
    m = 0
    n = 0

    # declaring variables used in MLP-Regressor
    hidden_layers = 40
    alpha = np.array([10**(-8), 10**(-7), 10**(-6), 10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1), 1, 10, 100])
    random_state = 10
    activation_mode = 'logistic'
    solver_mode = 'lbfgs'
    max_iter = 200
    train_mse = np.zeros((alpha.size, random_state))
    test_mse = np.zeros((alpha.size, random_state))

    for m in range(alpha.size):
        for n in range(random_state):

            nn = MLPRegressor(hidden_layer_sizes = (hidden_layers,), activation = activation_mode, solver = solver_mode, 
                              alpha = alpha[m], max_iter = max_iter, random_state = n)
            nn.fit(x_train, y_train)
            train_mse[m][n] = calculate_mse(nn, x_train, y_train)
            test_mse[m][n] = calculate_mse(nn, x_test, y_test)


    plot_mse_vs_alpha(train_mse, test_mse, alpha)

    pass


def ex_1_2_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 b)
    Remember to set alpha and momentum to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    ## TODO
    m = 0
    n = 0

    # declaring variables used in MLP-Regressor
    hidden_layers = 40
    alpha = 10**(-3)
    random_state = 10
    activation_mode = 'logistic'
    solver_mode = 'lbfgs'
    max_iter = 2000

    # declaring arrays to save the last iteration, the validation minimum and the test minimum
    err_last_iter = []
    err_valid_min = []
    err_test_min = []

    # same for test/train/valid mse
    test_mse = np.zeros((max_iter // 20, random_state))
    train_mse = np.zeros((max_iter // 20, random_state))
    valid_mse = np.zeros((max_iter // 20, random_state))

    # calculating the permutation
    permutation = np.random.permutation(x_train.size)

    # declaring new matrices for training and validation
    x_training_new = np.zeros((x_train.size // 2, 1))
    y_training_new = np.zeros(x_train.size // 2)

    x_validation_new = np.zeros((x_train.size // 2, 1))
    y_validation_new = np.zeros(x_train.size // 2)


    # fill the new training and validation set using the permutation defined above
    for m in range(x_train.size // 2):
        x_training_new[m] = x_train[permutation[m]]
        y_training_new[m] = y_train[permutation[m]]

        x_validation_new[m] = x_train[permutation[m + (x_train.size // 2)]]
        y_validation_new[m] = y_train[permutation[m + (y_train.size // 2)]]

    m = 0    

    for m in range(random_state):
        nn = MLPRegressor(hidden_layer_sizes = (hidden_layers,), activation = activation_mode, solver = solver_mode, 
                          alpha = alpha, warm_start = True, max_iter = max_iter, random_state = m)
        for n in range(max_iter):

            # monitor the error every 20 iterations
            if n % 20 == 0:
                nn.fit(x_training_new, y_training_new)
                test_mse[n // 20][m] = calculate_mse(nn, x_test, y_test)
                train_mse[n // 20][m] = calculate_mse(nn, x_training_new, y_training_new)
                valid_mse[n // 20][m] = calculate_mse(nn, x_validation_new, y_validation_new)

        err_last_iter.append(test_mse[((n + 1) // 20) - 1][m])
        err_valid_min.append(test_mse[np.argmin(valid_mse, axis = 0)[m]][m])
        err_test_min.append(np.min(test_mse, axis = 0)[m])

    plot_bars_early_stopping_mse_comparison(err_last_iter, err_valid_min, err_test_min)

    pass


def ex_1_2_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 c)
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    ## TODO
    pass
