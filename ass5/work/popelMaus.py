# Filename: HW5_skeleton.py
# Author: Christian Knoll
# Edited: May, 2018

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.stats import multivariate_normal
import pdb

import sklearn
from sklearn import datasets


# --------------------------------------------------------------------------------
# Assignment 5
def main():
    # ------------------------
    # 0) Get the input 
    # (a) load the modified iris data
    data, labels = load_iris_data()

    # (b) construct the datasets
    x_2dim = data[:, [0, 2]]

    #bonus
    # x_2dim = data[:, [0, 3]]

    x_4dim = data

    # TODO: implement PCA
    x_2dim_pca = PCA(data, nr_dimensions=2, whitening=False)

    # (c) visually inspect the data with the provided function (see example below)
    #plot_iris_data(x_2dim, labels)

    # ------------------------
    # 1) Consider a 2-dim slice of the data and evaluate the EM- and the KMeans- Algorithm   
    scenario = 1
    dim = 2
    nr_components = 3

    # TODO set parameters
    tol = 1e-6  # tolerance
    max_iter = 100  # maximum iterations for GN

    # TODO: implement
    (alpha_0, mean_0, cov_0) = init_EM(x_2dim, dimension=dim, nr_components=nr_components, scenario=scenario)

    alpha, mean, cov, ll, labels = EM(x_2dim, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)
    plt.plot(ll)
    plt.title("Loglikelihood over iterations (Scenario 1)")
    plt.xlabel("iteration")
    plt.ylabel("loglikelihood")
    plt.show()
    plotEM(mean, x_2dim, labels, scenario)
    for i in range(0, len(mean[0])):
        plot_gauss_contour(mean[:, i], cov[i], np.min(x_2dim[labels == i, 0]), np.max(x_2dim[labels == i, 0]), np.min(x_2dim[labels == i, 1]),
                           np.max(x_2dim[labels == i, 1]), len(x_2dim[labels == i]), "<3")
    plt.title("EM classification (Scenario 1)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    initial_centers = init_k_means(dimension=dim, nr_clusters=nr_components, scenario=scenario)
    print(initial_centers)
    centers, cum_dists, ls = k_means(x_2dim, nr_components, initial_centers, max_iter, tol)
    plotKmeans(centers, x_2dim, ls, "K-means result (Scenario 1)", scenario)
    plt.plot(cum_dists)
    plt.title("Cumulative distance (Scenario 1)")
    plt.xlabel("iteration")
    plt.ylabel("cumulative distance")
    plt.show()

    # TODO visualize your results

    # ------------------------
    # 2) Consider 4-dimensional data and evaluate the EM- and the KMeans- Algorithm 
    scenario = 2
    dim = 4
    nr_components = 3

    # TODO set parameters
    tol = 1e-6  # tolerance
    max_iter = 100  # maximum iterations for GN

    # TODO: implement
    (alpha_0, mean_0, cov_0) = init_EM(x_4dim, dimension=dim, nr_components=nr_components, scenario=scenario)

    alpha, mean, cov, ll, labels = EM(x_4dim, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)
    plt.plot(ll)
    plt.title("Loglikelihood over iterations (Scenario 2)")
    plt.xlabel("iteration")
    plt.ylabel("loglikelihood")
    plt.show()
    plotEM(mean, x_4dim, labels, scenario)

    for i in range(0, len(mean[0])):
        plot_gauss_contour([mean[:, i][0], mean[:, i][2]],
                           [[cov[i][0, 0], cov[i][0, 2]], [cov[i][2, 0], cov[i][2, 2]]],
                           np.min(x_4dim[labels == i, 0]),
                           np.max(x_4dim[labels == i, 0]),
                           np.min(x_4dim[labels == i, 2]),
                           np.max(x_4dim[labels == i, 2]),
                           len(x_4dim[labels == i]), "<3")

    plt.title("EM classification (Scenario 2)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    initial_centers = init_k_means(dimension=dim, nr_clusters=nr_components, scenario=scenario)
    centers, cum_dists, ls = k_means(x_4dim, nr_components, initial_centers, max_iter, tol)
    plotKmeans(centers, x_4dim, ls, "K-means result (Scenario 2)", scenario)
    plt.plot(cum_dists)
    plt.title("Cumulative distance (Scenario 2)")
    plt.xlabel("iteration")
    plt.ylabel("cumulative distance")
    plt.show()

    # TODO: visualize your results by looking at the same slice as in 1)

    # ------------------------
    # 3) Perform PCA to reduce the dimension to 2 while preserving most of the variance.
    # Then, evaluate the EM- and the KMeans- Algorithm  on the transformed data
    scenario = 3
    dim = 2
    nr_components = 4

    # TODO set parameters
    tol = 1e-6  # tolerance
    max_iter = 100  # maximum iterations for GN

    # TODO: implement
    (alpha_0, mean_0, cov_0) = init_EM(x_2dim_pca, dimension=dim, nr_components=nr_components, scenario=scenario)

    alpha, mean, cov, ll, labels = EM(x_2dim_pca, nr_components, alpha_0, mean_0, cov_0, max_iter, tol)
    plt.plot(ll)
    plt.title("Loglikelihood over iterations (Scenario 3)")
    plt.xlabel("iteration")
    plt.ylabel("loglikelihood")
    plt.show()
    plotEM(mean, x_2dim_pca, labels, scenario)
    for i in range(0, len(mean[0])):
        plot_gauss_contour(mean[:, i], cov[i], np.min(x_2dim_pca[labels == i, 0]), np.max(x_2dim_pca[labels == i, 0]),
                           np.min(x_2dim_pca[labels == i, 1]),
                           np.max(x_2dim_pca[labels == i, 1]), len(x_2dim_pca[labels == i]), "<3")
    plt.title("EM classification (Scenario 3)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    initial_centers = init_k_means(dimension=dim, nr_clusters=nr_components, scenario=scenario)
    centers, cum_dists, ls = k_means(x_2dim_pca, nr_components, initial_centers, max_iter, tol)
    plotKmeans(centers, x_2dim_pca, ls, "K-means result (Scenario 3)", scenario)
    plt.plot(cum_dists)
    plt.title("Cumulative distance (Scenario 3)")
    plt.xlabel("iteration")
    plt.ylabel("cumulative distance")
    plt.show()

    # TODO: visualize your results
    # TODO: compare PCA as pre-processing (3.) to PCA as post-processing (after 2.)

    pdb.set_trace()
    pass


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
def init_EM(data, dimension=2, nr_components=3, scenario=None):
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
    alpha_0 = np.array([1.0/float(nr_components) for i in range(0, nr_components)])
    mean_0 = np.transpose(np.array([data[np.random.randint(len(data), size=1)[0]] for d in range(0, nr_components)]))
    print(mean_0)
    cov_0 = np.array([3 * np.identity(dimension) for i in range(0, nr_components)])  # , (1, 2, 0))
    return alpha_0, mean_0, cov_0
    pass


# --------------------------------------------------------------------------------
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
    D = X.shape[1]
    assert D == mean_0.shape[0]
    N = X.shape[0]
    # TODO: iteratively compute the posterior and update the parameters
    alpha = alpha_0
    mean = mean_0
    cov = cov_0
    r = np.zeros((K, N))
    ll = 0
    lls = []
    for iteration in range(0, max_iter):
        # expect
        for k in range(0, K):
            for n in range(0, N):
                r[k][n] = e_step(k, alpha, mean, cov, X[n], K)
        # maxi
        cov = np.zeros(cov.shape)
        mean = np.zeros(mean.shape)
        for k in range(0, K):
            N_k = np.sum(r[k])
            for n in range(0, N):
                mean[:, k] += r[k][n] * X[n]
            mean[:, k] /= N_k
            for n in range(0, N):
                cov[k] += r[k][n] * np.dot((X[n] - mean[:, k])[:, None], (X[n] - mean[:, k])[None, :])
            cov[k] /= N_k
            alpha[k] = N_k / N
        # check for convergence
        new_ll = 0
        for n in range(0, N):
            ll_of_k = 0
            for k in range(0, K):
                ll_of_k += alpha[k] * normal(cov[k], mean[:, k], X[n])
            new_ll += np.log(ll_of_k)
        lls.append(new_ll)
        if np.abs(new_ll - ll) < tol:
            ll = new_ll
            break
        ll = new_ll


    # TODO: classify all samples after convergence
    labels = np.zeros((N,))
    for n in range(0, N):
        labels[n] = np.argmax(r[:, n])
    return alpha, mean, cov, lls, labels
    pass


def e_step(k, alpha, mean, cov, x, K):
    p_k = alpha[k] * normal(cov[k], mean[:, k], x)
    p_sum = 0
    for k_i in range(0, K):
        p_sum += alpha[k_i] * normal(cov[k_i], mean[:, k_i], x)
    return p_k / p_sum
    pass


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
    # plt.show()

# --------------------------------------------------------------------------------
def init_k_means(dimension=None, nr_clusters=None, scenario=None):
    """ initializes the k_means algorithm
    Input: 
        dimension... dimension D of the dataset, scalar
        nr_clusters...scalar
        scenario... optional parameter that allows to further specify the settings, scalar
    Returns:
        initial_centers... initial cluster centers,  D x nr_clusters"""
    # TODO chosse suitable inital values for each scenario

    return np.array([np.random.random_integers(3, 7, nr_clusters) for d in range(0, dimension)])
    pass


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


# --------------------------------------------------------------------------------
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
    Y = [[] for i in range(0, K)]
    centers = centers_0
    new_centers = centers
    cum_dists = []
    cum_dist = 0
    last_cum_dist = 0
    k_cum_dists = np.zeros((K, ))
    labels = np.zeros((N, ))

    for i in range(0, max_iter + 1):
        centers = new_centers
        cum_dist = 0
        k_cum_dists = np.zeros((K, ))
        for n in range(0, N):
            dist = np.zeros((K,))
            for k in range(0, K):
                dist[k] = np.dot((X[n] - centers[:, k]).T, X[n] - centers[:, k])
            Y[np.argmin(dist)].append(X[n])
            labels[n] = np.argmin(dist)
            k_cum_dists[np.argmin(dist)] += dist[np.argmin(dist)]
        cum_dist = sum(k_cum_dists)
        cum_dists.append(cum_dist)
        if np.abs(cum_dist - last_cum_dist) < tol:
            return centers, cum_dists, labels

        last_cum_dist = cum_dist

        new_centers = np.zeros(centers.shape)
        for k in range(0, K):
            if len(Y[k]) > 0:
                new_centers[:, k] = sum(Y[k]) / len(Y[k])
            else:
                new_centers[:, k] = np.random.random_integers(3, 7, D)
    # TODO: classify all samples after convergence
    return centers, cum_dists, labels
    pass

# --------------------------------------------------------------------------------
def PCA(data, nr_dimensions=None, whitening=False):
    """ perform PCA and reduce the dimension of the data (D) to nr_dimensions
    Input:
        data... samples, nr_samples x D
        nr_dimensions... dimension after the transformation, scalar
        whitening... False -> standard PCA, True -> PCA with whitening
        
    Returns:
        transformed data... nr_samples x nr_dimensions
        variance_explained... amount of variance explained by the the first nr_dimensions principal components, scalar"""
    if nr_dimensions is not None:
        dim = nr_dimensions
    else:
        dim = 2

    # TODO: Estimate the principal components and transform the data
    # using the first nr_dimensions principal_components
    S = np.cov(np.transpose(data))
    eigval, eigvec = np.linalg.eig(S)
    sorted_eigs = sorted(zip(eigval, eigvec), key=lambda tpl: tpl[0])
    U = [i[1] for i in sorted_eigs[:-dim]]
    U = np.array(U)

    new_data = np.zeros((data.shape[0], dim))
    for i in range(0, data.shape[0]):
        new_data[i] = np.dot(U, data[i])

    return new_data

    # TODO: Have a look at the associated eigenvalues and compute the amount of varianced explained


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------
def load_iris_data():
    """ loads and modifies the iris data-set
    Input: 
    Returns:
        X... samples, 150x4
        Y... labels, 150x1"""
    iris = datasets.load_iris()
    X = iris.data
    X[50:100, 2] = iris.data[50:100, 2] - 0.25
    Y = iris.target
    return X, Y


# --------------------------------------------------------------------------------
def plot_iris_data(data, labels):
    """ plots a 2-dim slice according to the specified labels
    Input:
        data...  samples, 150x2
        labels...labels, 150x1"""
    plt.scatter(data[labels == 0, 0], data[labels == 0, 1], label='Iris-Setosa')
    plt.scatter(data[labels == 1, 0], data[labels == 1, 1], label='Iris-Versicolor')
    plt.scatter(data[labels == 2, 0], data[labels == 2, 1], label='Iris-Virgnica')

    plt.legend()
    # plt.show()


# --------------------------------------------------------------------------------
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


def normal(cov, mean, x):
    cov = cov + (np.identity(cov.shape[0]) * 1e-8)
    p = (1.0 / ((2 * np.pi) ** (x.shape[0] / 2) * (np.linalg.det(cov) ** (1.0 / 2.0))))
    p *= np.exp(-(1.0 / 2.0) * np.dot(np.dot(np.transpose(x - mean), np.linalg.inv(cov)), (x - mean)))
    return p
# --------------------------------------------------------------------------------
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
    # plt.show()
    return


# --------------------------------------------------------------------------------
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
    cumulativePM = np.cumsum(PM)  # build CDF based on PMF
    offsetRand = np.random.uniform(0, 1) * (1 / N)  # offset to circumvent numerical issues with cumulativePM
    comb = np.arange(offsetRand, 1 + offsetRand, 1 / N)  # new axis with N values in the range ]0,1[

    j = 0
    for i in range(0, N):
        while comb[i] >= cumulativePM[j]:  # map the linear distributed values comb according to the CDF
            j += 1
        y[i] = X[j]

    return np.random.permutation(y)  # permutation of all samples


# --------------------------------------------------------------------------------
def reassign_class_labels(labels):
    """ reassigns the class labels in order to make the result comparable. 
    new_labels contains the labels that can be compared to the provided data,
    i.e., new_labels[i] = j means that i corresponds to j.
    Input:
        labels... estimated labels, 150x1
    Returns:
        new_labels... 3x1"""
    class_assignments = np.array([[np.sum(labels[0:50] == 0), np.sum(labels[0:50] == 1), np.sum(labels[0:50] == 2)],
                                  [np.sum(labels[50:100] == 0), np.sum(labels[50:100] == 1),
                                   np.sum(labels[50:100] == 2)],
                                  [np.sum(labels[100:150] == 0), np.sum(labels[100:150] == 1),
                                   np.sum(labels[100:150] == 2)]])
    new_labels = np.array([np.argmax(class_assignments[:, 0]),
                           np.argmax(class_assignments[:, 1]),
                           np.argmax(class_assignments[:, 2])])
    return new_labels


# --------------------------------------------------------------------------------
def sanity_checks():
    # likelihood_multivariate_normal
    mu = [0.0, 0.0]
    cov = [[1, 0.2], [0.2, 0.5]]
    x = np.array([[0.9, 1.2], [0.8, 0.8], [0.1, 1.0]])
    P = likelihood_multivariate_normal(x, mu, cov)
    print(P)

    # plot_gauss_contour(mu, cov, -2, 2, -2, 2, 100, 'Gaussian')
    # plt.show()

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
    reshuffled_labels[class_labels_unordererd == 0] = new_labels[0]
    reshuffled_labels[class_labels_unordererd == 1] = new_labels[1]
    reshuffled_labels[class_labels_unordererd == 2] = new_labels[2]


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    sanity_checks()
    main()
