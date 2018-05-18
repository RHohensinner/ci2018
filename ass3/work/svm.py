import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix

from svm_plot import plot_svm_decision_boundary, plot_score_vs_degree, plot_score_vs_gamma, plot_mnist, \
    plot_confusion_matrix

"""
Computational Intelligence TU - Graz
Assignment 3: Support Vector Machine, Kernels & Multiclass classification
Part 1: SVM, Kernels

TODOS are all contained here.
"""

__author__ = 'bellec,subramoney'


def ex_1_a(x, y):
    """
    Solution for exercise 1 a)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel
    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########

    # init linear svm
    kernel_mode = 'linear'
    lin_svm = svm.SVC(kernel = kernel_mode)
    # train linear svm
    lin_svm.fit(x, y)

    # plotting the decision boundary
    plot_svm_decision_boundary(lin_svm, x, y)


def ex_1_b(x, y):
    """
    Solution for exercise 1 b)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel
    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########
    # adding point (4, 0) with label 1
    x_new = np.vstack((x, [4, 0]))
    y_new = np.hstack((y, 1))

    # init linear svm
    kernel_mode = 'linear'
    lin_svm = svm.SVC(kernel = kernel_mode)

    # train linear svm
    lin_svm.fit(x_new, y_new)

    # plotting the decision boundary
    plot_svm_decision_boundary(lin_svm, x_new, y_new)


def ex_1_c(x, y):
    """
    Solution for exercise 1 c)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel with different values of C
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########

    # helper vars
    m = 0
    kernel_mode = 'linear'

    # given C values
    Cs = [1e6, 1, 0.1, 0.001]

    # adding point (4, 0) with label 1
    x_new = np.vstack((x, [4, -4]))
    y_new = np.hstack((y, 1))

    # for loop over all Cs values
    for m in range(len(Cs)):
        # init linear svm
        lin_svm = svm.SVC(kernel = kernel_mode, C = Cs[m])

        # train linear svm
        lin_svm.fit(x_new, y_new)

        # plotting the decision boundary
        plot_svm_decision_boundary(lin_svm, x_new, y_new)


def ex_2_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel for the given dataset
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########
    
    # init linear svm
    kernel_mode = 'linear'
    lin_svm = svm.SVC(kernel = kernel_mode)
    # train linear svm
    lin_svm.fit(x_train, y_train)

    # calc. & print svc.score() = mean accuracy of classification
    lin_svm_score = lin_svm.score(x_test, y_test)
    print("lin_swm_score: ", lin_svm_score)

    # plotting the decision boundary
    plot_svm_decision_boundary(lin_svm, x_train, y_train, x_test, y_test)


def ex_2_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with polynomial kernels for different values of the degree
    ## (Remember to set the 'coef0' parameter to 1)
    ## and plot the variation of the test and training scores with polynomial degree using 'plot_score_vs_degree' func.
    ## Plot the decision boundary and support vectors for the best value of degree
    ## using 'plot_svm_decision_boundary' function
    ###########

    # given degree values
    degrees = range(1, 20)

    # helper variables
    m = 0
    coef_value = 1
    kernel_mode = 'poly'

    all_train_scores = []
    temp_train = 0
    all_test_scores = []
    temp_test = 0
    temp_highscore = 0
    top_degree = 0

    # for loop over all 20 degrees, saving all results inbetween
    for m in degrees:
    	# init non-linear svm
        nonlin_svm = svm.SVC(kernel = kernel_mode, coef0 = coef_value, degree = m)
        # train non-linear svm
        nonlin_svm.fit(x_train, y_train)
        # calc scores
        temp_train = nonlin_svm.score(x_train, y_train)
        temp_test = nonlin_svm.score(x_test, y_test)
        # update highscore
        if(temp_test > temp_highscore):
        	temp_highscore = temp_test
        	top_degree = m
        # save scores
        all_train_scores.append(temp_train)
        all_test_scores.append(temp_test)

    #
    print("top degree: ", top_degree)
    print("top score: ", temp_highscore)

    # plotting scores
    plot_score_vs_degree(all_train_scores, all_test_scores, degrees)

    # recreate best svm and plotting it
    best_svm = svm.SVC(kernel = kernel_mode, coef0 = coef_value, degree = top_degree)
    best_svm.fit(x_train, y_train)
    plot_svm_decision_boundary(best_svm, x_train, y_train, x_test, y_test)





def ex_2_c(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 c)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with RBF kernels for different values of the gamma
    ## and plot the variation of the test and training scores with gamma using 'plot_score_vs_gamma' function.
    ## Plot the decision boundary and support vectors for the best value of gamma
    ## using 'plot_svm_decision_boundary' function
    ###########

    # given gamma values
    gammas = np.arange(0.01, 2, 0.02)

    # helper variables
    m = 0
    coef_value = 1
    kernel_mode = 'rbf'

    all_train_scores = []
    temp_train = 0
    all_test_scores = []
    temp_test = 0
    temp_highscore = 0
    top_gamma = 0

    # for loop over all 20 degrees, saving all results inbetween
    for m in gammas:
    	# init non-linear svm
        rbf_svm = svm.SVC(kernel = kernel_mode, gamma = m)
        # train non-linear svm
        rbf_svm.fit(x_train, y_train)
        # calc scores
        temp_train = rbf_svm.score(x_train, y_train)
        temp_test = rbf_svm.score(x_test, y_test)
        # update highscore
        if(temp_test > temp_highscore):
        	temp_highscore = temp_test
        	top_gamma = m
        # save scores
        all_train_scores.append(temp_train)
        all_test_scores.append(temp_test)

    #
    print("top gamma: ", top_gamma)
    print("top score: ", temp_highscore)

    # plotting scores
    plot_score_vs_gamma(all_train_scores, all_test_scores, gammas)

    # recreate best svm and plotting it
    best_svm = svm.SVC(kernel = kernel_mode, gamma = top_gamma)
    best_svm.fit(x_train, y_train)
    plot_svm_decision_boundary(best_svm, x_train, y_train, x_test, y_test)


def ex_3_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with one-versus-rest strategy with
    ## - linear kernel
    ## - rbf kernel with gamma going from 10**-5 to 10**5
    ## - plot the scores with varying gamma using the function plot_score_versus_gamma
    ## - Mind that the chance level is not .5 anymore and add the score obtained with the linear kernel as optional argument of this function
    ###########

    # helper variables
    c = 10
    kernel_mode_lin = 'linear'
    kernel_mode_rbf = 'rbf'
    df_shape = 'ovr'
    #gammas = np.arange(10**(-5), 10**(5), 20000)
    gammas = [10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3), 10**(4), 10**(5)]
    #gammas2 = np.linspace(10**(-5), 10**(5), 10)

    # init linear svm and train
    lin_svm = svm.SVC(kernel = kernel_mode_lin, C = c, decision_function_shape = df_shape)
    lin_svm.fit(x_train, y_train)

    # calc lin scores
    lin_trainscore = lin_svm.score(x_train, y_train)
    lin_testscore = lin_svm.score(x_test, y_test)

    print("LINEAR: \n", "trainscore: ", lin_trainscore, "testscore: ", lin_testscore)

    # init rbf svm and train it looping over gammas

    rbf_svm = svm.SVC(kernel = kernel_mode_rbf, C = c, decision_function_shape = df_shape)
    rbf_trainscore = []
    temp_train = 0
    rbf_testscore = []
    temp_test = 0

    for m in gammas:
        # setting current gamma
        rbf_svm.gamma = m

        # training
        rbf_svm.fit(x_train, y_train)

        # calc scores
        temp_train = rbf_svm.score(x_train, y_train)
        temp_test = rbf_svm.score(x_test, y_test)

        # save scores
        rbf_trainscore.append(temp_train)
        rbf_testscore.append(temp_test)

    print("RBF: \n", "trainscore: ",max(rbf_trainscore), "testscore: ", max(rbf_testscore))

    plot_score_vs_gamma(rbf_trainscore, rbf_testscore, gammas, lin_trainscore, lin_testscore, 0.2)

def ex_3_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with a LINEAR kernel
    ## Use the sklearn.metrics.confusion_matrix to plot the confusion matrix.
    ## Find the index for which you get the highest error rate.
    ## Plot the confusion matrix with plot_confusion_matrix.
    ## Plot the first 10 occurrences of the most misclassified digit using plot_mnist.
    ###########

    # helper variables
    m = 0
    c = 10
    kernel_mode = 'linear'

    # init linear svm and train it
    lin_svm = svm.SVC(kernel = kernel_mode, C = c)
    lin_svm.fit(x_train, y_train)

    # pred y to plot conf matrix
    y_pred= lin_svm.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plot_confusion_matrix(cm, lin_svm.classes_)

    # helper variables
    most_misclassified_number = 0
    temp_m = cm[0][0]

	# searching for the most missclassifed number/label
    for m in range(1, 5):
        if(temp_m > cm[m][m]):
            temp_m = cm[m][m]
            most_misclassified_number = m

    # given labels
    labels = range(1, 6)

    # helper variables
    temp_list = []
    image_counter = 0
    max_pred = len(y_pred)
    m = 0

    # getting indices of missclassified numbers
    for m in range(0, max_pred):
        if(labels[most_misclassified_number] == y_pred[m]):
            if(y_test[m] != y_pred[m]):
            	# add the missclassified image-index to the list
                temp_list.append(m)
                image_counter = image_counter + 1
                # if we have 10 images stop
                if(image_counter == 10):
                    break

    # given output/plot --------------------------------------------------------------------------------

    # Numpy indices to select images that are misclassified.
    sel_err = np.array(temp_list)
    # should be the label number corresponding the largest classification error
    i = most_misclassified_number

    # Plot with mnist plot
    plot_mnist(x_test[sel_err], y_pred[sel_err], labels=labels[i], k_plots=10, prefix='Predicted class')
