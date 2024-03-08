import pandas as pd
import numpy as np
from scipy.stats import norm, t
from scipy.optimize import minimize
import statsmodels.api as sm
from non_psd_fixes import near_psd, chol_psd

'''
Multivariate Normal Simulation
'''


def simulateNormal(N, df, mean=None, seed=1234, fixMethod=near_psd):
    # Error Checking
    m, n = df.shape
    if n != m:
        raise ValueError(f"Covariance Matrix is not square ({n},{m})")

    # Initialize the output
    out = np.zeros((N, n))

    # Set mean
    if mean is None:
        mean = np.zeros(n)
    else:
        if len(mean) != n:
            raise ValueError(f"Mean ({len(mean)}) is not the size of cov ({n},{n})")

    # Set the seed to make sure the value is the same each time.
    np.random.seed(seed)

    eigenvalues, eigenvectors = np.linalg.eig(df)
    # If the covariance is not PSD, try to fix it
    if min(eigenvalues) < 0:
        df = fixMethod(df)

    # Take the root (cholesky factorization)
    l = chol_psd(df)

    # Generate random standard normals
    rand_normals = np.random.normal(0.0, 1.0, size=(N, n))

    # Apply the Cholesky root and plus the mean to the random normals
    out = np.dot(rand_normals, l.T) + mean

    return out.T


'''
Multivariate PCA Simulation
'''


def simulatePCA(N, df, mean=None, seed=1234, pctExp=1):
    # Error Checking
    m, n = df.shape
    if n != m:
        raise ValueError(f"Covariance Matrix is not square ({n},{m})")

    # Initialize the output
    out = np.zeros((N, n))

    # Set mean
    if mean is None:
        mean = np.zeros(n)
    else:
        if len(mean) != n:
            raise ValueError(f"Mean ({len(mean)}) is not the size of cov ({n},{n})")

    eigenvalues, eigenvectors = np.linalg.eig(df)

    # Get the indices that would sort eigenvalues in descending order
    indices = np.argsort(eigenvalues)[::-1]
    # Sort eigenvalues
    eigenvalues = eigenvalues[indices]
    # Sort eigenvectors according to the same order
    eigenvectors = eigenvectors[:, indices]

    tv = np.sum(eigenvalues)
    posv = np.where(eigenvalues >= 1e-8)[0]
    if pctExp <= 1:
        nval = 0
        pct = 0.0
        # How many factors needed
        for i in posv:
            pct += eigenvalues[i] / tv
            nval += 1
            if pct >= pctExp:
                break

    # If nval is less than the number of positive eigenvalues, truncate posv
    if nval < len(posv):
        posv = posv[:nval]

    # Filter eigenvalues based on posv
    eigenvalues = eigenvalues[posv]
    eigenvectors = eigenvectors[:, posv]

    B = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    np.random.seed(seed)
    rand_normals = np.random.normal(0.0, 1.0, size=(N, len(posv)))
    out = np.dot(rand_normals, B.T) + mean

    return out.T


'''
Calculate Return
'''


# Implement the function to calculate the return
def return_calculate(prices, method='ARS', dateColumn='Date'):
    # Exclude the date column from the calculations
    tickers = [col for col in prices.columns if col != dateColumn]
    df = prices[tickers]  # The dataframe is now with no date column.

    # Calculate the return using Classical Brownian Motion.
    if method == 'CBM':
        df = df.diff().dropna()

    # Calculate the return using Arithmetic Return System.
    elif method == 'ARS':
        df = (df - df.shift(1)) / df.shift(1)
        df = df.dropna()

    # Calculate the return using Geometric Brownian Motion.
    elif method == 'GBM':
        df = np.log(df).diff().dropna()

    else:
        raise ValueError(f"method: {method} must be in (\"CBM\",\"ARS\",\"GBM\")")

    return df


'''
Fit the Data with Normal Distribution
'''


def fit_normal(data):
    # Fit the normal distribution to the data
    mu, std = norm.fit(data)
    return mu, std


'''
Fit the Data with t Distribution
'''


def fit_general_t(data):
    # Fit the t distribution to the data
    nu, mu, sigma = t.fit(data)
    return mu, sigma, nu


'''
Fit the Data with t Distribution - regression
'''


def fit_regression_t(df):
    Y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    betas = MLE_t(X, Y)
    X = sm.add_constant(X)

    # Get the residuals.
    e = Y - np.dot(X, betas)

    params = t.fit(e)
    out = {"mu": [params[1]],
           "sigma": [params[2]],
           "nu": [params[0]]}
    for i in range(len(betas)):
        out["B" + str(i)] = betas[i]
    out = pd.DataFrame(out)
    out.rename(columns={'B0': 'Alpha'}, inplace=True)
    return out


# The objective negative log-likelihood function (need to be minimized).
def MLE_t(X, Y):
    X = sm.add_constant(X)

    def ll_t(params):
        nu, sigma = params[:2]
        beta_MLE_t = params[2:]
        epsilon = Y - np.dot(X, beta_MLE_t)
        # Calculate the log-likelihood
        log_likelihood = np.sum(t.logpdf(epsilon, df=nu, loc=mu, scale=sigma))
        return -log_likelihood

    beta = np.zeros(X.shape[1])
    nu, mu, sigma = 1, 0, np.std(Y - np.dot(X, beta))
    params = np.append([nu, sigma], beta)
    bnds = ((1e-9, None), (1e-9, None), (None, None), (None, None), (None, None), (None, None))

    # Minimize the log-likelihood to get the beta
    res = minimize(ll_t, params, bounds=bnds, options={'disp': True})
    beta_MLE = res.x[2:]
    return beta_MLE

