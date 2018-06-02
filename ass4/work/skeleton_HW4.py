#Filename: HW4_skeleton.py
#Author: Christian Knoll, Florian Kaum
#Edited: May, 2018

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats

import math as mt

from scipy.stats import multivariate_normal

#--------------------------------------------------------------------------------
# Assignment 4
def main():
    
    # choose the scenario
#    scenario = 1    # all anchors are Gaussian
#    scenario = 2    # 1 anchor is exponential, 3 are Gaussian
    scenario = 3    # all anchors are exponential
    
    # specify position of anchors
    p_anchor = np.array([[5,5],[-5,5],[-5,-5],[5,-5]])
    nr_anchors = np.size(p_anchor,0)
    
    # position of the agent for the reference mearsurement
    p_ref = np.array([[0,0]])
    # true position of the agent (has to be estimated)
    p_true = np.array([[2,-4]])
#    p_true = np.array([[2,-4])
     
    #show init state:                   
    #plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref)
    
    # load measured data and reference measurements for the chosen scenario
    data,reference_measurement = load_data(scenario)
    
    # get the number of measurements 
    assert(np.size(data,0) == np.size(reference_measurement,0))
    nr_samples = np.size(data,0)
    
    #1) ML estimation of model parameters
    # DONE
    params = parameter_estimation(reference_measurement, nr_anchors, p_anchor, p_ref, p_true, data, scenario)
    
    #2) Position estimation using least squares
    # DONE
    position_estimation_least_squares(data, nr_anchors, p_anchor, p_true, True, scenario)

    if(scenario == 3):
        # TODO: don't forget to plot joint-likelihood function for the first measurement

        #3) Postion estimation using numerical maximum likelihood
        # TODO
        position_estimation_numerical_ml(data, nr_anchors, p_anchor, params, p_true)
    
        #4) Position estimation with prior knowledge (we roughly know where to expect the agent)
        # TODO
        # specify the prior distribution
        prior_mean = p_true
        prior_cov = np.eye(2)
        position_estimation_bayes(data, nr_anchors, p_anchor, prior_mean, prior_cov, params, p_true)

    pass

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def parameter_estimation(reference_measurement, nr_anchors, p_anchor, p_ref, p_true,data, scenario):
    """ estimate the model parameters for all 4 anchors based on the reference measurements, i.e., for anchor i consider reference_measurement[:,i]
    Input:
        reference_measurement... nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2
        p_ref... reference point, 2x2 """
    params = np.zeros([1, nr_anchors])

    #TODO (1) check whether a given anchor is Gaussian or exponential
    
    # check visually if anchor's are Gaussian or Exponential
    if(scenario == 2):
        find_out_exp_distr(reference_measurement)



    #TODO (2) estimate the according parameter based

    # calc sigmas/lamdas
    sigmas = calc_sigma_squared(data, p_anchor, p_true[0])
    lamdas = calc_lambda(data, p_anchor, p_true[0])

    if(scenario == 1):
        print("Scenario 1: Sigmas - ", sigmas)
        params[0][0] = sigmas[0]
        params[0][1] = sigmas[1]
        params[0][2] = sigmas[2]
        params[0][3] = sigmas[3]

    elif(scenario == 2):
        print("Scenario 2: Lamda -", lamdas[0])
        print("Scenario 2: Sigmas -", sigmas[1], sigmas[2], sigmas[3])
        params[0][0] = lamdas[0]
        params[0][1] = sigmas[1]
        params[0][2] = sigmas[2]
        params[0][3] = sigmas[3]

    elif(scenario == 3):
        print("Scenario 3: Lamda - ", lamdas)
        params[0][0] = lamdas[0]
        params[0][1] = lamdas[1]
        params[0][2] = lamdas[2]
        params[0][3] = lamdas[3]

    return params
#--------------------------------------------------------------------------------
def position_estimation_least_squares(data, nr_anchors, p_anchor, p_true, use_exponential, scenario):
    """estimate the position by using the least squares approximation. 
    Input:
        data...distance measurements to unkown agent, nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2 
        p_true... true position (needed to calculate error) 2x2 
        use_exponential... determines if the exponential anchor in scenario 2 is used, bool"""
    nr_samples = np.size(data,0)
    p_est = np.zeros((nr_samples, 2))
    err_measures = np.zeros(nr_samples)
    
    #TODO set parameters
    #tol = ...  # tolerance
    #max_iter = ...  # maximum iterations for GN
    tol = 10e-5
    max_iter = 50
    
    
    if(scenario == 1):
        title = "Scenario 1: Gauss"
        print(title)
    
    elif(scenario == 2):
        title = "Scenario 2: Gauss and Exponential"
        print(title)

    elif(scenario == 3):
        title = "Scenario 3: Exponential"
        print(title)



    # TODO estimate position for  i in range(0, nr_samples)

    for i in range(0, nr_samples):
        r = data[i]
        p_start = -5.0 + 10.0 * np.random.rand(1, 2)
        p_est[i] = least_squares_GN(p_anchor, p_start, r, max_iter, tol)

	# TODO calculate error measures and create plots

    for i in range(0, nr_samples):
        diff = np.subtract(p_est[i], p_true)[0]
        err_measures[i] = mt.sqrt(mt.pow(diff[0], 2) + mt.pow(diff[1], 2))

    # printing mean error and variance
    print("Mean: ", np.mean(err_measures), "Variance: ", np.var(err_measures))

    # split x,y estimated and calculate their mean vector and covariance matrix
    x_est = p_est[:, 0]
    y_est  = p_est[:, 1]

    x_mean = np.mean(x_est)
    y_mean = np.mean(y_est)

    mean_vector = [x_mean, y_mean]

    cov_mat = np.cov(x_est, y_est)

    # Scatter plots of estimated positions:
    plt.axis([-6, 6, -6, 6])
    for i in range(0, 4):
        plt.plot(p_anchor[i, 0], p_anchor[i, 1], 'bo')
        plt.text(p_anchor[i, 0] + 0.2, p_anchor[i, 1] + 0.2, r'$p_{a,' + str(i) + '}$')
    plt.plot(p_true[0, 0], p_true[0, 1], 'r*')
    plt.text(p_true[0, 0] + 0.2, p_true[0, 1] + 0.2, r'$p_{true}$')
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    plt.scatter(x_est, y_est, 0.5)

    # setting up vars for plotGaussianContour
    min_x = min(x_est)
    max_x = max(x_est)
    min_y = min(y_est)
    max_y = max(y_est)

    # plotting gaussian contour
    plot_gauss_contour(mean_vector, cov_mat, min_x, max_x, min_y, max_y, title)

    # cumulative distribution function
    Fx, x = ecdf(err_measures)
    plt.plot(x, Fx)
    plt.title("CDF - Scenario " + str(scenario))
    plt.xlabel("position of est_errors")
    plt.ylabel("P(est_errors)")
    plt.show()

#--------------------------------------------------------------------------------
def position_estimation_numerical_ml(data,nr_anchors,p_anchor, lambdas, p_true):
    """ estimate the position by using a numerical maximum likelihood estimator
    Input:
        data...distance measurements to unkown agent, nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2 
        lambdas... estimated parameters (scenario 3), nr_anchors x 1
        p_true... true position (needed to calculate error), 2x2 """
    #TODO
    pass
#--------------------------------------------------------------------------------
def position_estimation_bayes(data,nr_anchors,p_anchor,prior_mean,prior_cov,lambdas, p_true):
    """ estimate the position by accounting for prior knowledge that is specified by a bivariate Gaussian
    Input:
         data...distance measurements to unkown agent, nr_measurements x nr_anchors
         nr_anchors... scalar
         p_anchor... position of anchors, nr_anchors x 2
         prior_mean... mean of the prior-distribution, 2x1
         prior_cov... covariance of the prior-dist, 2x2
         lambdas... estimated parameters (scenario 3), nr_anchors x 1
         p_true... true position (needed to calculate error), 2x2 """
    # TODO
    pass
#--------------------------------------------------------------------------------
def least_squares_GN(p_anchor, p_start, r, max_iter, tol):
    """ apply Gauss Newton to find the least squares solution
    Input:
        p_anchor... position of anchors, nr_anchors x 2
        p_start... initial position, 2x1
        r... distance_estimate, nr_anchors x 1
        max_iter... maximum number of iterations, scalar
        tol... tolerance value to terminate, scalar"""

    # helper vars:    
    p_temp = p_start
    p_old = 0
    brk_cond = 0
    diff = 0


    for i in range(max_iter):
        # creating jacobi matrix
        j_mat = calc_jacobi(p_temp, p_anchor)

        #updating p_old and calc new temp
        p_old = p_temp
        anchor_dists = calc_anchor_dist(p_temp[0], p_anchor)
        p_temp = np.subtract(p_temp, np.dot(np.dot(np.linalg.inv(np.dot(j_mat.T, j_mat)), j_mat.T), np.subtract(r, anchor_dists)))
        
        # calc diff and break condition
        diff = np.subtract(p_temp, p_old)[0]
        brk_cond = mt.sqrt(mt.pow(diff[0], 2) + mt.pow(diff[1], 2))

        # if brk_cond is smaller than tol break and return p_temp
        if (brk_cond < tol):
            break;

    return p_temp;

    
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Helper Functions
#--------------------------------------------------------------------------------

# 2 Maximum Likelihood Estimation of Model Parameters:
def calc_euclid_dist(a, b):
    distance = mt.sqrt(mt.pow((a[0] - b[0]), 2) + mt.pow((a[1] - b[1]), 2))
    return distance

#--------------------------------------------------------------------------------
def find_out_exp_distr(data):
    for i in range(4):
        plt.hist(data[:, i])
        plt.show()

#--------------------------------------------------------------------------------
def calc_mean(a, b):
    return calc_euclid_dist(a, b)

#--------------------------------------------------------------------------------
def calc_sigma_squared(data, p_anchor, p_true):
    
    sigmas_squared = []

    for i in range(4):
        N = len(data)
        total_sigma = 0

        for j in range(N):
            total_sigma += mt.pow((data[j][i] - calc_mean(p_anchor[i], p_true)), 2)

        sigmas_squared.append(total_sigma / N)

    return sigmas_squared

#--------------------------------------------------------------------------------
def calc_lambda(data, p_anchor, p_true):
    
    lamdas = []

    for i in range(4):
        N = len(data)
        current_lamda = N / (np.sum(data[:,i]) - calc_mean(p_true, p_anchor[i]) * N)
        lamdas.append(current_lamda)

    return lamdas

#--------------------------------------------------------------------------------

# 3 Estimation of the Position:
def calc_jacobi(p_start, p_anchor):
    j_mat = np.zeros((4, 2))

    for i in range(4):
        j_mat[i][0] = ((p_anchor[i][0] - p_start[0][0]) / calc_euclid_dist(p_start[0], p_anchor[i]))
        j_mat[i][1] = ((p_anchor[i][1] - p_start[0][1]) / calc_euclid_dist(p_start[0], p_anchor[i]))

    return j_mat

#--------------------------------------------------------------------------------
def calc_anchor_dist(p_true, p_anchor):
    dists = []
    for i in range(4):
        dists.append(calc_euclid_dist(p_true, p_anchor[i]))
    return dists

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def plot_gauss_contour(mu,cov,xmin,xmax,ymin,ymax,title="Title"):
    
    """ creates a contour plot for a bivariate gaussian distribution with specified parameters
    
    Input:
      mu... mean vector, 2x1
      cov...covariance matrix, 2x2
      xmin,xmax... minimum and maximum value for width of plot-area, scalar
      ymin,ymax....minimum and maximum value for height of plot-area, scalar
      title... title of the plot (optional), string"""
    
	#npts = 100
    delta = 0.025
    x = np.arange(xmin, xmax, delta)
    y = np.arange(ymin, ymax, delta)
    X, Y = np.meshgrid(x, y)
    Z = mlab.bivariate_normal(X,Y,np.sqrt(cov[0][0]),np.sqrt(cov[1][1]),mu[0], mu[1], cov[0][1])
    plt.plot([mu[0]],[mu[1]],'r+') # plot the mean as a single point
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(title)
    plt.show()
    return

#--------------------------------------------------------------------------------
def ecdf(realizations):   
    """ computes the empirical cumulative distribution function for a given set of realizations.
    The output can be plotted by plt.plot(x,Fx)
    
    Input:
      realizations... vector with realizations, Nx1
    Output:
      x... x-axis, Nx1
      Fx...cumulative distribution for x, Nx1"""
    x = np.sort(realizations)
    Fx = np.linspace(0,1,len(realizations))
    return Fx,x

#--------------------------------------------------------------------------------
def load_data(scenario):
    """ loads the provided data for the specified scenario
    Input:
        scenario... scalar
    Output:
        data... contains the actual measurements, nr_measurements x nr_anchors
        reference.... contains the reference measurements, nr_measurements x nr_anchors"""
    data_file = 'measurements_' + str(scenario) + '.data'
    ref_file =  'reference_' + str(scenario) + '.data'
    
    data = np.loadtxt(data_file,skiprows = 0)
    reference = np.loadtxt(ref_file,skiprows = 0)
    
    return (data,reference)

#--------------------------------------------------------------------------------
def plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref=None):
    """ plots all anchors and agents
    Input:
        nr_anchors...scalar
        p_anchor...positions of anchors, nr_anchors x 2
        p_true... true position of the agent, 2x1
        p_ref(optional)... position for reference_measurements, 2x1"""
    # plot anchors and true position
    plt.axis([-6, 6, -6, 6])
    for i in range(0, nr_anchors):
        plt.plot(p_anchor[i, 0], p_anchor[i, 1], 'bo')
        plt.text(p_anchor[i, 0] + 0.2, p_anchor[i, 1] + 0.2, r'$p_{a,' + str(i) + '}$')
    plt.plot(p_true[0, 0], p_true[0, 1], 'r*')
    plt.text(p_true[0, 0] + 0.2, p_true[0, 1] + 0.2, r'$p_{true}$')
    if p_ref is not None:
        plt.plot(p_ref[0, 0], p_ref[0, 1], 'r*')
        plt.text(p_ref[0, 0] + 0.2, p_ref[0, 1] + 0.2, '$p_{ref}$')
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    plt.show()
    pass

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
