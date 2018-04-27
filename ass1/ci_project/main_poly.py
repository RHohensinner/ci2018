#!/usr/bin/env python
import json
import matplotlib.pyplot as plt
import poly
from plot_poly import plot_poly
from plot_poly import plot_errors
import numpy as np

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Linear Regression with polynomial features

This files:
1) loads the data from 'data_linreg.json'
2) trains and test a linear regression model for a given number of degree
3) plots the results

TODO boxes are to be found in 'poly.py'
"""


def main():
    # Set the degree of the polynomial expansion

    degree = 1
    degrees = []

    for j in range(30):
        degrees.append(j)

    for i in range(1, 31):

        degree = i

        data_path = 'data_linreg.json'

        # Load the data
        f = open(data_path, 'r')
        data = json.load(f)
        for k, v in data.items():
            data[k] = np.array(v).reshape((len(v), 1))

        # Print the training and testing errors
        theta, err_train, err_val, err_test = poly.train_and_test(data, degree)
        print('Training error {} \t Validation error {} \t Testing error {} '.format(err_train, err_val, err_test))
        
        if(degree == 1):
            lowest = err_train

        if(lowest > err_train):
            lowest = err_train
            lowest_degree = degree
            lowest_theta = theta


#def plot_poly(data, degree, theta_opt, n_line_precision=100):
#def plot_errors(i_best, degrees, mse_train, mse_val, mse_test):
    plot_errors(13, degrees, err_train, err_val, err_test)
    plt.show()


if __name__ == '__main__':
    main()
