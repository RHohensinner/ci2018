#Filename: HW5_skeleton.py
#Author: Christian Knoll
#Edited: May, 2018

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.stats import multivariate_normal
import pdb

import sklearn
from sklearn import datasets
#--------------------------------------------------------------------------------
# Assignment 5
def main():   
    #------------------------
    # 0) Get the input 
    ## (a) load the modified iris data
    data, labels = load_iris_data()
    
    ## (b) construct the datasets
    #scenario 1
    x_2dim = data[:, [0,2]]
    x_2dim_bonus = data[:, [0,3]]
    #scenario 2
    x_4dim = data
    #scenario 3
    x_2dim_pca, variance_explained = PCA(data, 2, False)
    print("variance explained " + str(variance_explained))
    
    ## (c) visually inspect the data with the provided function (see example below)
    plot_iris_data(x_2dim, labels)

    #------------------------
    # SCENARIO 1:
    # 1) Consider a 2-dim slice of the data and evaluate the EM- and the KMeans- Algorithm   
    scenario = 1
    dim = 2
    nr_components = 3
    
    #TODO set parameters
    tol = 0.001  # tolerance
    max_iter = 60  # maximum iterations for GN
    
    #TODO: implement
    #(alpha_0, mean_0, cov_0) = init_EM(dimension = dim, nr_components= nr_components, scenario=scenario)
    #... = EM(x_2dim,nr_components, alpha_0, mean_0, cov_0, max_iter, tol)    
    #initial_centers = init_k_means(dimension = dim, nr_clusters=nr_components, scenario=scenario)
    #... = k_means(x_2dim, nr_components, initial_centers, max_iter, tol)

    alpha_0, mean_0, cov_0 = init_EM(dim, nr_components, scenario, x_2dim)
    alpha, mean, cov, likelihood, label_names = EM(x_2dim, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)  

    plt.plot(likelihood)
    plt.title("loglikelihood for Scenario 1")
    plt.xlabel("it")
    plt.ylabel("llh")
    plt.show()
   
    for k in range(nr_components):
        plt.scatter(x_2dim[label_names == k, 0], x_2dim[label_names == k, 1], label='component ' + str(k))
    plt.scatter(mean[0], mean[1], label='mu')
    plt.legend()

    for k in range(nr_components):
        plot_gauss_contour(mean[:, k], 
                           cov[k], 
                           np.min(x_2dim[label_names == k, 0]), 
                           np.max(x_2dim[label_names == k, 0]), 
                           np.min(x_2dim[label_names == k, 1]),
                           np.max(x_2dim[label_names == k, 1]), 
                           len(x_2dim[label_names == k]), "Expectation Maximization Algorithm")
    plt.title("Expectation Maximization Algorithm for Scenario 1")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()

    #------------------------------------------------------------------------------------------------------

    initial_centers = init_k_means(dim, nr_components, scenario, x_2dim)
    centers, cumulative_distance, label_names = k_means(x_2dim, nr_components, initial_centers, max_iter, tol)
    
    
    for k in range(nr_components):
        plt.scatter(x_2dim[label_names == k, 0], x_2dim[label_names == k, 1], label='component ' + str(k))
    plt.scatter(centers[0], centers[1], label='mu')


    plt.legend()
    plt.title("K-Means Algorithm for Scenario 1")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()

    plt.plot(cumulative_distance)
    plt.title("Cumulative distance for Scenario 1")
    plt.xlabel("it")
    plt.ylabel("cum dist")
    plt.show()

    """
    #------------------------
    # 2) Consider 4-dimensional data and evaluate the EM- and the KMeans- Algorithm 
    scenario = 2
    dim = 4
    nr_components = 3
    
    #TODO set parameters
    #tol = ...  # tolerance
    #max_iter = ...  # maximum iterations for GN
    #nr_components = ... #n number of components
    
    #TODO: implement
    #(alpha_0, mean_0, cov_0) = init_EM(dimension = dim, nr_components= nr_components, scenario=scenario)
    #... = EM(x_4dim, nr_components, alpha_0, mean_0, cov_0, max_iter, tol) 
    #initial_centers = init_k_means(dimension = dim, nr_cluster=nr_components, scenario=scenario)
    #... = k_means(x_4dim,nr_components, initial_centers, max_iter, tol)

    alpha_0, mean_0, cov_0 = init_EM(dim, nr_components, scenario, x_4dim)
    alpha, mean, cov, likelihood, label_names = EM(x_4dim, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)  

    plt.plot(likelihood)
    plt.title("loglikelihood for Scenario 2")
    plt.xlabel("it")
    plt.ylabel("llh")
    plt.show()
   
    for k in range(nr_components):
        plt.scatter(x_4dim[label_names == k, 0], x_4dim[label_names == k, 2], label='component ' + str(k))
    plt.scatter(mean[0], mean[1], label='mu')
    plt.legend()

    print(label_names)
    for k in range(nr_components):
        plot_gauss_contour([mean[:, k][0], mean[:, k][2]], 
                           [[cov[k][0,0], cov[k][0,2]],[cov[k][2,0], cov[k][2,2]]], 
                           np.min(x_4dim[label_names == k, 0]), 
                           np.max(x_4dim[label_names == k, 0]), 
                           np.min(x_4dim[label_names == k, 2]),
                           np.max(x_4dim[label_names == k, 2]), 
                           len(x_4dim[label_names == k]), "Expectation Maximization Algorithm")
    plt.title("Expectation Maximization Algorithm for Scenario 2")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()

    #------------------------------------------------------------------------------------------------------

    initial_centers = init_k_means(dim, nr_components, scenario, x_4dim)
    centers, cumulative_distance, label_names = k_means(x_4dim, nr_components, initial_centers, max_iter, tol)
    
    
    for k in range(nr_components):
        plt.scatter(x_4dim[label_names == k, 0], x_4dim[label_names == k, 2], label='component ' + str(k))
    plt.scatter(centers[0], centers[1], label='mu')


    plt.legend()
    plt.title("K-Means Algorithm for Scenario 2")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()

    plt.plot(cumulative_distance)
    plt.title("Cumulative distance for Scenario 2")
    plt.xlabel("it")
    plt.ylabel("cum dist")
    plt.show()
    
    
    #TODO: visualize your results by looking at the same slice as in 1)"""
    
    
    #------------------------
    # 3) Perform PCA to reduce the dimension to 2 while preserving most of the variance.
    # Then, evaluate the EM- and the KMeans- Algorithm  on the transformed data
    scenario = 3
    dim = 2
    nr_components = 3
    
    #TODO set parameters
    #tol = ...  # tolerance
    #max_iter = ...  # maximum iterations for GN
    #nr_components = ... #n number of components
    
    #TODO: implement
    alpha_0, mean_0, cov_0 = init_EM(dim, nr_components, scenario, x_2dim_pca)
    alpha, mean, cov, likelihood, label_names = EM(x_2dim_pca, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)  

    plt.plot(likelihood)
    plt.title("loglikelihood for Scenario 1")
    plt.xlabel("it")
    plt.ylabel("llh")
    plt.show()
   
    for k in range(nr_components):
        plt.scatter(x_2dim_pca[label_names == k, 0], x_2dim_pca[label_names == k, 1], label='component ' + str(k))
    plt.scatter(mean[0], mean[1], label='mu')
    plt.legend()

    for k in range(nr_components):
        plot_gauss_contour(mean[:, k], 
                           cov[k], 
                           np.min(x_2dim_pca[label_names == k, 0]), 
                           np.max(x_2dim_pca[label_names == k, 0]), 
                           np.min(x_2dim_pca[label_names == k, 1]),
                           np.max(x_2dim_pca[label_names == k, 1]), 
                           len(x_2dim_pca[label_names == k]), "Expectation Maximization Algorithm")
    plt.title("Expectation Maximization Algorithm for Scenario 1")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()

    #------------------------------------------------------------------------------------------------------

    initial_centers = init_k_means(dim, nr_components, scenario, x_2dim_pca)
    centers, cumulative_distance, label_names = k_means(x_2dim_pca, nr_components, initial_centers, max_iter, tol)
    
    
    for k in range(nr_components):
        plt.scatter(x_2dim_pca[label_names == k, 0], x_2dim_pca[label_names == k, 1], label='component ' + str(k))
    plt.scatter(centers[0], centers[1], label='mu')


    plt.legend()
    plt.title("K-Means Algorithm for Scenario 3 (PCA)")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()

    plt.plot(cumulative_distance)
    plt.title("Cumulative distance for Scenario 3 (PCA)")
    plt.xlabel("it")
    plt.ylabel("cum dist")
    plt.show()
    
    #TODO: visualize your results
    #TODO: compare PCA as pre-processing (3.) to PCA as post-processing (after 2.)
    
    exit()
    pdb.set_trace()
    pass

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def init_EM(dimension = 2, nr_components = 3, scenario = None, data = None):
    """ initializes the EM algorithm
    Input: 
        dimension... dimension D of the dataset, scalar
        nr_components...scalar
        scenario... optional parameter that allows to further specify the settings, scalar
    Returns:
        alpha_0... initial weight of each component, 1 x nr_components
        mean_0 ... initial mean values, D x nr_components
        cov_0 ...  initial covariance for each component, D x D x nr_components"""

    # TODO choose suitable initial values for each scenario

    # 1st) INIT:
    #mean_good_temp = [5.0, 1.5, 6.0, 3.75, 6.75, 6.0]
    #mean_bad_temp = [6.1, 4.5, 6.2, 4.5, 6.3, 4.5]

    nr = 0
    dim = 0
    N = data.shape[0]

    # setting alpha_0 (see script page 18)
    alpha_0 = np.ones(nr_components)
    mean_0 = np.zeros((dimension, nr_components))
    cov_0 = np.zeros((nr_components, dimension, dimension))

    alpha_0 = alpha_0 * (1.0 / float(nr_components))
    #print(alpha_0)

    # setting mean_0 (see script page 19)
    mean_counter = 0
    for nr in range(nr_components):
        for dim in range(dimension):
            r = np.random.randint(0, N)
            mean_0[dim][nr] = data[r][0]
            #mean_0[dim][nr] = mean_bad_temp[mean_counter]

            #mean_counter = mean_counter + 1
    print(mean_0)


    # setting cov_0 (see script page 19)
    counter = 0
    temp_mean = (1.0 / N * sum(data))
    diff = (data - temp_mean)
    # idea from: 
    # https://stackoverflow.com/a/33641428
    diff_transpose = np.transpose(diff)
    # attention, dimensions are reversed!
    temp_cov = (1.0 / N * np.matmul(diff_transpose, diff))

    for counter in range(nr_components):
        cov_0[counter] = temp_cov
    #print(cov_0)

    #mu_good =[ [5.0, 1.5], [6.0, 3.75], [6.75, 6.0]]

    #mu_good =[ [5.0, 6.0, 6.75], [1.5, 3.75, 6.0]]
    #mean_0 = mu_good

    return alpha_0, mean_0, cov_0

    """
    #print(dimension, nr_components, scenario)
    counter = 0

    n_count = data.shape[0]

    
    # attention, dimensions are reversed!
    alpha_0 = np.ones(nr_components)
    mean_0 = np.zeros((nr_components, dimension))
    cov_0 = np.zeros((nr_components, dimension, dimension))


	# setting alpha_0 (see script page 18)
    alpha_0 = alpha_0 * (1.0 / nr_components)

    # setting mean_0 (see script page 19)
    for counter in range(nr_components):
        r = np.random.randint(0, n_count)
        mean_0[counter] = data[r]

    # setting cov_0 (see script page 19)
    counter = 0
    temp_mean = (1.0 / n_count * sum(data))
    diff = (data - temp_mean)
    # idea from: 
    # https://stackoverflow.com/a/33641428
    diff_transpose = np.transpose(diff)
    # attention, dimensions are reversed!
    temp_cov = (1.0 / n_count * np.matmul(diff_transpose, diff))

    for counter in range(nr_components):
    	cov_0[counter] = temp_cov


    return alpha_0, mean_0, cov_0

    
    M = nr_components

    alpha_0 = []
    for i in range(M):
        alpha_0.append(1/M)
    return alpha_0
    
    cov_0 = []
    x = X[:,0]
    y = X[:,1]
    for i in range(M):
        cov_0.append(np.cov(x, y))

    mean_0 = X[rd.randint(0, np.size(X, 0), size = M)]

    return alpha_0, mean_0, cov_0    

    """

#--------------------------------------------------------------------------------
def EM(X, K, alpha_0, mean_0, cov_0, max_iter, tol):
    """ perform the EM-algorithm in order to optimize the parameters of a GMM
    with K components
    Input:
        X... samples, nr_samples x dimension (D)
        K... nr of components, scalar
        alpha_0... initial weight of each component, 1 x K
        mean_0 ... initial mean values, D x K
        cov_0 ...  initial covariance for each component, D x D x K        
    Returns:
        alpha... final weight of each component, 1 x K
        mean...  final mean values, D x K
        cov...   final covariance for ech component, D x D x K
        log_likelihood... log-likelihood over all iterations, nr_iterations x 1
        labels... class labels after performing soft classification, nr_samples x 1"""
    # compute the dimension 
    #print(X.shape[1], mean_0.shape[1])
    D = X.shape[1]
    assert D == mean_0.shape[0]

    #TODO: iteratively compute the posterior and update the parameters
    #TODO: classify all samples after convergence

    alpha = alpha_0
    mean = mean_0
    cov = cov_0

    N = X.shape[0] 

    r_n_k = np.zeros((K, N))

    log_lhs = []
    log_lh = 0


    for iteration in range(max_iter):

        # --------------------------------------------------------------------------------
        # 2nd) Expectation Step:
        # --------------------------------------------------------------------------------

        for n in range(N):
            lh_sum = 0
            for k in range(K):
                # to prevent singular matrix
                lh_sum += alpha[k] * likelihood_multivariate_normal(X[n], mean[:, k], cov[k] + (np.identity(cov[k].shape[0]) * 1e-5)) 
            for k in range(K):
                lh = likelihood_multivariate_normal(X[n], mean[:, k], cov[k] + (np.identity(cov[k].shape[0]) * 1e-5))
                lh_numerator = alpha[k] * lh
                lh_denominator = lh_sum

                r_n_k[k][n] = (lh_numerator / lh_denominator)
                
        # --------------------------------------------------------------------------------
        # 3rd) Maximization Step:
        # --------------------------------------------------------------------------------        
        
        cov = np.zeros(cov.shape)
        mean = np.zeros(mean.shape)

        for k in range(K):
            # equals Nm in script
            N_k = np.sum(r_n_k[k])
            part_sum = 0
            for n in range(N):
                part_sum += r_n_k[k][n] * X[n]
            mean[:, k] = part_sum / N_k

            part_sum = 0 
            for n in range(N):
                diff = X[n] - mean[:,k]
                # in order to get a DxD-matrix from 2 vectors
                part_sum += r_n_k[k][n] * np.dot((diff)[:, None], (diff)[None, :])

            cov[k] = part_sum / N_k

            alpha[k] = N_k / N

        # --------------------------------------------------------------------------------
        # 4th) Likelihood Calculation:
        # --------------------------------------------------------------------------------

        curr_lh = 0
        for n in range(N):
            curr_lh_k = 0
            for k in range(K):
                curr_lh_k += alpha[k] * likelihood_multivariate_normal(X[n], mean[:, k], cov[k] + (np.identity(cov[k].shape[0]) * 1e-5), True) 
            curr_lh += np.log(np.abs(curr_lh_k))

        log_lhs.append(curr_lh)
        
        abs_var = np.abs(curr_lh - log_lh)
        log_lh = curr_lh

        if abs_var < tol:
            break

    label_names = np.zeros((N,))
    for n in range(N):
        label_names[n] = np.argmax(r_n_k[:, n])

    return alpha, mean, cov, log_lhs, label_names 


def plotEM(means, data, labels, scenario):

    if scenario == 1 or scenario == 3:
        for i in range(0, len(means[0])):
            plt.scatter(data[labels == i, 0], data[labels == i, 1], label='class ' + str(i))
        plt.scatter(means[0], means[1], label='Means')

    if scenario == 2:
        for i in range(0, len(means[0])):
            plt.scatter(data[labels == i, 0], data[labels == i, 2], label='class ' + str(i))
        plt.scatter(means[0], means[2], label='Means')

    plt.legend()
    #plt.show()

#--------------------------------------------------------------------------------
def init_k_means(dimension=None, nr_clusters=None, scenario=None, data = None):
    """ initializes the k_means algorithm
    Input: 
        dimension... dimension D of the dataset, scalar
        nr_clusters...scalar
        scenario... optional parameter that allows to further specify the settings, scalar
    Returns:
        initial_centers... initial cluster centers,  D x nr_clusters"""
    # TODO chosse suitable inital values for each scenario

    N = data.shape[0]

    init_array = np.zeros((dimension, nr_clusters))
    for d in range(dimension):
        for n in range(nr_clusters):
            r = np.random.randint(0, N)
            init_array[d][n] = data[r][0]

    print(init_array)
    return init_array
#--------------------------------------------------------------------------------
def k_means(X, K, centers_0, max_iter, tol):
    """ perform the KMeans-algorithm in order to cluster the data into K clusters
    Input:
        X... samples, nr_samples x dimension (D)
        K... nr of clusters, scalar
        centers_0... initial cluster centers,  D x nr_clusters
    Returns:
        centers... final centers, D x nr_clusters
        cumulative_distance... cumulative distance over all iterations, nr_iterations x 1
        labels... class labels after performing hard classification, nr_samples x 1"""
    D = X.shape[1]
    assert D == centers_0.shape[0]
    # TODO: iteratively update the cluster centers
    N = X.shape[0]    

    centers = centers_0    
    cumulative_distance = []
    labels = np.zeros((N,))

    cumulative_distance_previous = 0

    Y = [[] for k in range(K)]

    m = 0;

    while(1):
        cumulative_distance_part = np.zeros((K,))
        cumulative_distance_current = 0
        
        for n in range(N):
            k_distance = np.zeros((K,))

            for k in range(0, K):
                diff = (X[n] - centers[:, k])
                k_distance[k] = np.dot(diff.T, diff)

            k_argmin = np.argmin(k_distance)

            labels[n] = k_argmin

            #add points that belong to that component
            Y[k_argmin].append(X[n])
            
            cumulative_distance_part[k_argmin] += k_distance[k_argmin]

        
        cumulative_distance_current = sum(cumulative_distance_part)

        cumulative_distance.append(cumulative_distance_current)

        cond_distance = cumulative_distance_previous - cumulative_distance_current

        m += 1
        if np.abs(cond_distance) < tol or m == max_iter:
            break

        for k in range(K):
            actual_member = len(Y[k])
            actual_sum = sum(Y[k])
            if actual_member == 0:
                centers[:, k] = init_k_means(D, K, data=X)[0]
            else:
                centers[:, k] = actual_sum / actual_member

        cumulative_distance_previous = cumulative_distance_current

    return centers, cumulative_distance, labels

def plotKmeans(centers, data, labels, title, scenario):
    if scenario == 1 or scenario == 3:
        for i in range(0, len(centers[0])):
            plt.scatter(data[labels == i, 0], data[labels == i, 1], label='class ' + str(i))
        plt.scatter(centers[0], centers[1], label='Means')

    if scenario == 2:
        for i in range(0, len(centers[0])):
            plt.scatter(data[labels == i, 0], data[labels == i, 2], label='class ' + str(i))
        plt.scatter(centers[0], centers[2], label='Means')

    plt.legend()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

#--------------------------------------------------------------------------------
def PCA(data, nr_dimensions=None, whitening=False):
    """ perform PCA and reduce the dimension of the data (D) to nr_dimensions
    Input:
        data... samples, nr_samples x D
        nr_dimensions... dimension after the transformation, scalar
        whitening... False -> standard PCA, True -> PCA with whitening
        
    Returns:
        transformed data... nr_samples x nr_dimensions
        variance_explained... amount of variance explained by the the first nr_dimensions principal components, scalar"""

    # firstly calculate covariance matrix of data
    # then calculate eigenvektors and eigenvalues
    # from D to M projection
    # search for eigenvectors with greatest variance
    if nr_dimensions is not None:
        dim = nr_dimensions
    else:
        dim = 2
    
    #TODO: Estimate the principal components and transform the data
    # using the first nr_dimensions principal_components

    cov_matrix = np.cov(np.transpose(data))
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix) # get eigenvector
    zipped_eigen = zip(eigenvalues, eigenvectors)
    sorted_list = sorted(zipped_eigen, key=lambda x: x[0]) # sort all x after 0th axis of this object
    print(sorted_list[0])
    M = dim
    l = len(sorted_list)
    big_u = [sorted_list[v][1] for v in range(l - M, l)]

    print(big_u)

    big_u_np_array = np.array(big_u)

    new_array = np.zeros((data.shape[0], M))
    for n in range(data.shape[0]):
        new_array[n] = np.dot(big_u_np_array, data[n])

    #explained
    eigenvalues_array = [sorted_list[v][0] for v in range(l - M, l)]
    explaination = sum(eigenvalues_array) / sum(eigenvalues)

    return new_array, explaination

    
    #TODO: Have a look at the associated eigenvalues and compute the amount of varianced explained
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Helper Functions
#--------------------------------------------------------------------------------
def load_iris_data():
    """ loads and modifies the iris data-set
    Input: 
    Returns:
        X... samples, 150x4
        Y... labels, 150x1"""
    iris = datasets.load_iris()
    X = iris.data
    X[50:100,2] =  iris.data[50:100,2]-0.25
    Y = iris.target    
    return X,Y   
#--------------------------------------------------------------------------------
def plot_iris_data(data,labels):
    """ plots a 2-dim slice according to the specified labels
    Input:
        data...  samples, 150x2
        labels...labels, 150x1"""
    plt.scatter(data[labels==0,0], data[labels==0,1], label='Iris-Setosa')
    plt.scatter(data[labels==1,0], data[labels==1,1], label='Iris-Versicolor')
    plt.scatter(data[labels==2,0], data[labels==2,1], label='Iris-Virgnica')
    
    plt.legend()
    plt.show()
#--------------------------------------------------------------------------------
def likelihood_multivariate_normal(X, mean, cov, log=False):
   """Returns the likelihood of X for multivariate (d-dimensional) Gaussian 
   specified with mu and cov.
   
   X  ... vector to be evaluated -- np.array([[x_00, x_01,...x_0d], ..., [x_n0, x_n1, ...x_nd]])
   mean ... mean -- [mu_1, mu_2,...,mu_d]
   cov ... covariance matrix -- np.array with (d x d)
   log ... False for likelihood, true for log-likelihood
   """

   dist = multivariate_normal(mean, cov)
   if log is False:
       P = dist.pdf(X)
   elif log is True:
       P = dist.logpdf(X)
   return P 

#--------------------------------------------------------------------------------
def plot_gauss_contour(mu, cov, xmin, xmax, ymin, ymax, nr_points, title="Title"):
    """ creates a contour plot for a bivariate gaussian distribution with specified parameters
    
    Input:
      mu... mean vector, 2x1
      cov...covariance matrix, 2x2
      xmin,xmax... minimum and maximum value for width of plot-area, scalar
      ymin,ymax....minimum and maximum value for height of plot-area, scalar
      nr_points...specifies the resolution along both axis
      title... title of the plot (optional), string"""

    # npts = 100
    delta_x = float(xmax - xmin) / float(nr_points)
    delta_y = float(ymax - ymin) / float(nr_points)
    x = np.arange(xmin, xmax, delta_x)
    y = np.arange(ymin, ymax, delta_y)

    X, Y = np.meshgrid(x, y)
    Z = mlab.bivariate_normal(X, Y, np.sqrt(cov[0][0]), np.sqrt(cov[1][1]), mu[0], mu[1], cov[0][1])
    plt.plot([mu[0]], [mu[1]], 'r+')  # plot the mean as a single point
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(title)
    #plt.show()
    return
#--------------------------------------------------------------------------------    
def sample_discrete_pmf(X, PM, N):
    """Draw N samples for the discrete probability mass function PM that is defined over 
    the support X.
       
    X ... Support of RV -- np.array([...])
    PM ... P(X) -- np.array([...])
    N ... number of samples -- scalar
    """
    assert np.isclose(np.sum(PM), 1.0)
    assert all(0.0 <= p <= 1.0 for p in PM)
    
    y = np.zeros(N)
    cumulativePM = np.cumsum(PM) # build CDF based on PMF
    offsetRand = np.random.uniform(0, 1) * (1 / N) # offset to circumvent numerical issues with cumulativePM
    comb = np.arange(offsetRand, 1 + offsetRand, 1 / N) # new axis with N values in the range ]0,1[
    
    j = 0
    for i in range(0, N):
        while comb[i] >= cumulativePM[j]: # map the linear distributed values comb according to the CDF
            j += 1	
        y[i] = X[j]
        
    return np.random.permutation(y) # permutation of all samples
#--------------------------------------------------------------------------------
def reassign_class_labels(labels):
    """ reassigns the class labels in order to make the result comparable. 
    new_labels contains the labels that can be compared to the provided data,
    i.e., new_labels[i] = j means that i corresponds to j.
    Input:
        labels... estimated labels, 150x1
    Returns:
        new_labels... 3x1"""
    class_assignments = np.array([[np.sum(labels[0:50]==0)   ,  np.sum(labels[0:50]==1)   , np.sum(labels[0:50]==2)   ],
                                  [np.sum(labels[50:100]==0) ,  np.sum(labels[50:100]==1) , np.sum(labels[50:100]==2) ],
                                  [np.sum(labels[100:150]==0),  np.sum(labels[100:150]==1), np.sum(labels[100:150]==2)]])
    new_labels = np.array([np.argmax(class_assignments[:,0]),
                           np.argmax(class_assignments[:,1]),
                           np.argmax(class_assignments[:,2])])
    return new_labels
#--------------------------------------------------------------------------------
def sanity_checks():
    # likelihood_multivariate_normal
    mu =  [0.0, 0.0]
    cov = [[1, 0.2],[0.2, 0.5]]
    x = np.array([[0.9, 1.2], [0.8, 0.8], [0.1, 1.0]])
    P = likelihood_multivariate_normal(x, mu, cov)
    print(P)
    
    plot_gauss_contour(mu, cov, -2, 2, -2, 2,100, 'Gaussian')

    # sample_discrete_pmf
    PM = np.array([0.2, 0.5, 0.2, 0.1])
    N = 1000
    X = np.array([1, 2, 3, 4])
    Y = sample_discrete_pmf(X, PM, N)
    
    print('Nr_1:', np.sum(Y == 1),
          'Nr_2:', np.sum(Y == 2),
          'Nr_3:', np.sum(Y == 3),
          'Nr_4:', np.sum(Y == 4))
    
    # re-assign labels
    class_labels_unordererd = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
       0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0,
       0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
       0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0])
    new_labels = reassign_class_labels(class_labels_unordererd)
    reshuffled_labels = np.zeros_like(class_labels_unordererd)
    reshuffled_labels[class_labels_unordererd==0] = new_labels[0]
    reshuffled_labels[class_labels_unordererd==1] = new_labels[1]
    reshuffled_labels[class_labels_unordererd==2] = new_labels[2]


    
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
if __name__ == '__main__':
    
    sanity_checks()
    main()
