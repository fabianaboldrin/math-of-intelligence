import numpy as np
import pandas as pd
from patsy import dmatrices
import warnings

def sigmoid(x):
    '''SIGMOID FUNCTION FOR X'''

    return 1/(1+np.exp(-x))

## Algorith settings

np.random.seed(0) # set the seed
tol=1e-8 # convergence tolerance
lam = None # l2 - regularization
max_iter = 20 # maximum allowed iterations
r = 0.95 # covariance between x and z
n = 1000 # number of observations (size of dataset to generate)
sigma = 1 # variance of noise - how spread out is the data?

## Model settings

beta_x, beta_z, beta_v = -4, .9, 1 # true beta coefficients
var_x, var_z, var_v = 1, 1, 4 # variances of inputs

## the model specification i want to fit
formula = 'y ~ x + z + v + np.exp(x) + I(v**2 + z)'

# keeping x and z closely related (height and weight)
x, z = np.random.multivariate_normal([0,0], [[var_x, r], [r, var_z]], n).T
# blood pressure
v = np.random.normal(0, var_v, n)**3

# create a pandas dataframe
A = pd.DataFrame({'x': x, 'z': z, 'v': v})

# compute the log offs for our 3 independent variables
# using the sigmoid function
A['log_odds'] = sigmoid(A[['x', 'z', 'v']].dot([beta_x, beta_z, beta_v]) + sigma * np.random.normal(0,1,n))

#compute te probability sample from binomial distribuition
A['y'] = [np.random.binomial(1,p) for p in A.log_odds]

# create a dataframe that encompasses our input data, model formula, and outputs
y, X = dmatrices(formula, A, return_type = 'dataframe')

X.head()

def catch_singularity(f):
    def silencer(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except np.linalg.LinAlgError:
            warnings.warn('Algotithm terminated - singular Hessian')
            return args[0]
        return silencer


def newton_step(curr, X, lam=None):
    p = np.array(sigmoid(X.dot(curr[:, 0])), ndmin=2).T
    w = np.diag((p * (1 - p))[:, 0])
    hessian = X.T.dot(w).dot(X)
    grad = X.T.dot(y - p)

    if lam:
        step, *_ = np.linalg.lstsq(hessian + lam * np.eye(curr.shape[0]), grad)
    else:
        step, *_ = np.linalg.lstsq(hessian, grad)

    beta = curr + step

    return beta


def alt_newton_step(curr, X, lam=None):
    p = np.array(sigmoid(X.dot(curr[:, 0])), ndmin=2).T
    w = np.diag((p * (1 - p))[:, 0])
    hessian = X.T.dot(w).dot(X)
    grad = X.T.dot(y - p)

    if lam:
        step = np.dot(np.linalg(hessian + lam * np.eye(curr.chape[0])), grad)
    else:
        step = np.dot(np.linalg.inv(hessian), grad)

    beta = curr + step

    return beta


def check_coefs_convergence(beta_old, beta_new, tol, iters):
    coef_change = np.abs(beta_old - beta_new)

    return not (np.any(coef_change > tol) & (iters < max_iter))

## initial conditions
# initial coefficients (weight values), 2 copies, we'll update one
beta_old, beta = np.ones((len(X.columns), 1)), np.zeros((len(X.columns), 1))

iter_count = 0

coefs_converged = False

while not coefs_converged:
    beta_old = beta
    beta = newton_step(beta, X, lam=lam)
    iter_count += 1
    coefs_converged = check_coefs_convergence(beta_old, beta, tol, iter_count)

print('Iterations: {}'.format(iter_count))
print('Beta: {}'.format(beta))