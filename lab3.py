""" CS5340 Lab 3: Hidden Markov Models
See accompanying PDF for instructions.

Name: Shubhankar Agrawal
Email: e0925482@u.nus.edu
Student ID: A0248330L
"""
import numpy as np
import scipy.stats
from scipy.special import softmax
from sklearn.cluster import KMeans

"""Helper Functions Begin Here"""
def forward_prop(x_list, pi, A, phi):
    """ Forward propagation procedure for the forward-backward algorithm.
    x_list (List[np.ndarray]): List of sequences of observed measurements
    pi (np.ndarray): Current estimated Initial state distribution (K,)
    A (np.ndarray): Current estimated Transition matrix (K, K)
    phi (Dict[np.ndarray]): Current estimated gaussian parameters
    """

    # Creating the empty lists for alpha, c and p_x_z
    n_states = pi.shape[0]
    alpha_list = [np.zeros([len(x), n_states]) for x in x_list]
    c_list = [np.zeros(len(x)) for x in x_list]
    p_x_z = [np.zeros([len(x), n_states]) for x in x_list]

    # Iterating through each sequence
    for i in range(len(x_list)):
        x = x_list[i]
        alpha = alpha_list[i]
        c = c_list[i]

        # Iterating through each observation and calculating emission probability
        for state in range(n_states):
            p_x_z[i][:, state] = scipy.stats.norm.pdf(x, phi['mu'][state], phi['sigma'][state])

        # Computing alpha values
        for n in range(len(x)):
            if n == 0:
                alpha[n] = pi * p_x_z[i][n]
            else:
                alpha[n] = np.dot(alpha[n-1], A) * p_x_z[i][n]

            # Scaling alpha values
            c[n] = np.sum(alpha[n])
            alpha[n] = alpha[n] / c[n]

    return alpha_list, c_list, p_x_z

def backward_prop(x_list, A, p_x_z, c_list):
    """ Backward propagation procedure for the forward-backward algorithm.
    x_list (List[np.ndarray]): List of sequences of observed measurements
    A (np.ndarray): Current estimated Transition matrix (K, K)
    p_x_z (List[np.ndarray]): List of probability of each observation given each state
    c_list (List[np.ndarray]): List of scaling factors for each sequence
    """

    # Creating the empty list for beta
    n_states = A.shape[0]
    beta_list = [np.zeros([len(x), n_states]) for x in x_list]

    # Iterating through each sequence
    for i in range(len(x_list)):
        x = x_list[i]
        beta = beta_list[i]
        c = c_list[i]

        # Computing beta values
        for n in range(len(x)-1, -1, -1):
            if n == len(x) - 1:
                beta[n] = np.ones(n_states)
            else:
                beta[n] = np.dot(A, p_x_z[i][n+1] * beta[n+1]) / c[n+1] #Scaling beta values

    return beta_list
"""Helper Functions End Here"""

def initialize(n_states, x):
    """Initializes starting value for initial state distribution pi
    and state transition matrix A.

    A and pi are initialized with random starting values which satisfies the
    summation and non-negativity constraints.
    """
    seed = 5340
    np.random.seed(seed)

    pi = np.random.random(n_states)
    A = np.random.random([n_states, n_states])

    # We use softmax to satisify the summation constraints. Since the random
    # values are small and similar in magnitude, the resulting values are close
    # to a uniform distribution with small jitter
    pi = softmax(pi)
    A = softmax(A, axis=-1)

    # Gaussian Observation model parameters
    # We use k-means clustering to initalize the parameters.
    x_cat = np.concatenate(x, axis=0)
    kmeans = KMeans(n_clusters=n_states, random_state=seed).fit(x_cat[:, None])
    mu = kmeans.cluster_centers_[:, 0]
    std = np.array([np.std(x_cat[kmeans.labels_ == l]) for l in range(n_states)])
    phi = {'mu': mu, 'sigma': std}

    return pi, A, phi


"""E-step"""
def e_step(x_list, pi, A, phi):
    """ E-step: Compute posterior distribution of the latent variables,
    p(Z|X, theta_old). Specifically, we compute
      1) gamma(z_n): Marginal posterior distribution, and
      2) xi(z_n-1, z_n): Joint posterior distribution of two successive
         latent states

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        pi (np.ndarray): Current estimated Initial state distribution (K,)
        A (np.ndarray): Current estimated Transition matrix (K, K)
        phi (Dict[np.ndarray]): Current estimated gaussian parameters

    Returns:
        gamma_list (List[np.ndarray]), xi_list (List[np.ndarray])
    """
    n_states = pi.shape[0]
    gamma_list = [np.zeros([len(x), n_states]) for x in x_list]
    xi_list = [np.zeros([len(x)-1, n_states, n_states]) for x in x_list]

    """ YOUR CODE HERE
    Use the forward-backward procedure on each input sequence to populate 
    "gamma_list" and "xi_list" with the correct values.
    Be sure to use the scaling factor for numerical stability.
    """
    # Forward and backward propagation to get alpha and beta values
    alpha, c, p_x_z = forward_prop(x_list, pi, A, phi)
    beta = backward_prop(x_list, A, p_x_z, c)

    for i in range(len(x_list)):
        x = x_list[i]

        # Computing gamma values
        gamma_list[i] = alpha[i] * beta[i]
        for n in range(len(x)):
            if n < len(x) - 1:

                # Computing xi values
                xi_list[i][n] = alpha[i][n][:, None] * A * p_x_z[i][n+1] * beta[i][n+1]
                xi_list[i][n] = xi_list[i][n] / np.sum(xi_list[i][n])

    return gamma_list, xi_list


"""M-step"""
def m_step(x_list, gamma_list, xi_list):
    """M-step of Baum-Welch: Maximises the log complete-data likelihood for
    Gaussian HMMs.
    
    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        gamma_list (List[np.ndarray]): Marginal posterior distribution
        xi_list (List[np.ndarray]): Joint posterior distribution of two
          successive latent states

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.
    """

    n_states = gamma_list[0].shape[1]
    pi = np.zeros([n_states])
    A = np.zeros([n_states, n_states])
    phi = {'mu': np.zeros(n_states),
           'sigma': np.zeros(n_states)}

    """ YOUR CODE HERE
    Compute the complete-data maximum likelihood estimates for pi, A, phi.
    """
    
    # Converting gamma, x and xi lists to numpy arrays
    gamma_np = np.array(gamma_list)
    xi_np = np.array(xi_list)
    x_np = np.array(x_list)

    for i in range(n_states):
        # Computing pi values
        pi[i] = np.sum(gamma_np[:, 0, i]) / np.sum(gamma_np[:, 0,:])

        # Computing phi values
        phi['mu'][i] = np.sum(gamma_np[:,:,i] * x_np) / np.sum(gamma_np[:,:,i])
        phi['sigma'][i] = np.sqrt(np.sum(gamma_np[:,:,i] * (x_np - phi['mu'][i])**2) / np.sum(gamma_np[:,:,i]))

        # Computing A values
        for j in range(len(A[i])):
            A[i][j] = np.sum(xi_np[:,:,i,j]) / np.sum(xi_np[:, :, i, :])



    return pi, A, phi


"""Putting them together"""
def fit_hmm(x_list, n_states):
    """Fit HMM parameters to observed data using Baum-Welch algorithm

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        n_states (int): Number of latent states to use for the estimation.

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Time-independent stochastic transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.

    """

    # We randomly initialize pi and A, and use k-means to initialize phi
    # Please do NOT change the initialization function since that will affect
    # grading
    pi, A, phi = initialize(n_states, x_list)

    """ YOUR CODE HERE
     Populate the values of pi, A, phi with the correct values. 
    """
    while(True):
        phi_old = phi
        gamma, xi = e_step(x_list, pi, A, phi)
        pi, A, phi = m_step(x_list, gamma, xi)
        if(np.all(np.abs(phi_old['mu'] - phi['mu']) < 1e-4) and np.all(np.abs(phi_old['sigma'] - phi['sigma']) < 1e-4)):
            break

    return pi, A, phi
