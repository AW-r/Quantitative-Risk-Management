import pandas as pd
import numpy as np
from scipy.stats import norm, t
from scipy.integrate import quad

'''
Fit the Data with Normal Distribution
'''


def fit_normal(data):
    # Fit the normal distribution to the data
    mu, std = norm.fit(data)
    return mu, std


'''
VaR for Normal Distribution
'''


def var_normal(data, alpha=0.05):
    # Fit the data with normal distribution.
    mu, std = fit_normal(data)

    # Calculate the VaR
    VaR = -norm.ppf(alpha, mu, std)

    # Calculate the relative difference from the mean expected.
    VaR_diff = VaR + mu

    return pd.DataFrame({"VaR Absolute": [VaR],
                         "VaR Diff from Mean": [VaR_diff]})


'''
ES for Normal Distribution
'''


def es_normal(data, alpha=0.05):
    # Fit the data with normal distribution.
    mu, std = fit_normal(data)

    # Calculate the VaR
    res = var_normal(data, alpha)
    VaR = res.iloc[0, 0]

    # Define the integrand function: x times the PDF of the distribution
    def integrand(x, mu, std):
        return x * norm.pdf(x, loc=mu, scale=std)

    # Calculate the ES
    ES, _ = quad(lambda x: integrand(x, mu, std), -np.inf, -VaR)
    ES /= -alpha

    # Calculate the relative difference from the mean expected.
    ES_diff = ES + mu

    return pd.DataFrame({"ES Absolute": [ES],
                         "ES Diff from Mean": [ES_diff]})


'''
Fit the Data with t Distribution
'''


def fit_general_t(data):
    # Fit the t distribution to the data
    nu, mu, sigma = t.fit(data)
    return mu, sigma, nu


'''
VaR for t Distribution
'''


def var_t(data, alpha=0.05):
    # Fit the data with t distribution.
    mu, sigma, nu = fit_general_t(data)

    # Calculate the VaR
    VaR = -t.ppf(alpha, nu, mu, sigma)

    # Calculate the relative difference from the mean expected.
    VaR_diff = VaR + mu

    return pd.DataFrame({"VaR Absolute": [VaR],
                         "VaR Diff from Mean": [VaR_diff]})


'''
ES for t Distribution
'''


def es_t(data, alpha=0.05):
    # Fit the data with normal distribution.
    mu, sigma, nu = fit_general_t(data)

    # Calculate the VaR
    res = var_t(data, alpha)
    VaR = res.iloc[0, 0]

    # Define the integrand function: x times the PDF of the distribution
    def integrand(x, mu, sigma, nu):
        return x * t.pdf(x, df=nu, loc=mu, scale=sigma)

    # Calculate the ES
    ES, _ = quad(lambda x: integrand(x, mu, sigma, nu), -np.inf, -VaR)
    ES /= -alpha

    # Calculate the relative difference from the mean expected.
    ES_diff = ES + mu

    return pd.DataFrame({"ES Absolute": [ES],
                         "ES Diff from Mean": [ES_diff]})


'''
VaR for t Distribution simulation
'''


def es_simulation(data, alpha=0.05, size=10000):
    # Fit the data with t distribution.
    mu, sigma, nu = fit_general_t(data)

    # Generate given size random numbers from a t-distribution
    random_numbers = t.rvs(df=nu, loc=mu, scale=sigma, size=size)

    return es_t(random_numbers, alpha)


'''
ES for Normal Distribution with an EW variance
'''


def es_ew(data, lmbd, alpha=0.05):
    # Calculate the mean.
    mu = data.mean()

    # Calculate the variance with an EW variance.
    std = np.sqrt(ew_cov_corr(data, lmbd, 'cov')).iloc[0, 0]
    data += mu

    # Calculate the VaR
    res = var_ew(data, lmbd, alpha)
    VaR = res.iloc[0, 0]

    # Define the integrand function: x times the PDF of the distribution
    def integrand(x, mu, std):
        return x * norm.pdf(x, loc=mu, scale=std)

    ES, _ = quad(lambda x: integrand(x, mu, std), -np.inf, -VaR)
    ES /= -alpha

    # Calculate the relative difference from the mean expected.
    ES_diff = ES + mu
    ES_diff = ES_diff[0]

    return pd.DataFrame({"ES Absolute": [ES],
                         "ES Diff from Mean": [ES_diff]})


'''
VaR and ES for historic simulation
'''


def historic(data, N=100000, alpha=0.05):
    # Use numpy's random.choice to draw N samples with replacement
    np.random.seed(50)
    simulated_draws = np.random.choice(data.iloc[:, 0], size=N, replace=True)

    # Calculate the mean for the data.
    mu = simulated_draws.mean()

    # Sorted the value in order to get the alpha% of the distribution
    sorted_data = np.sort(simulated_draws)

    # Calculate the percentage and dollar basis VaR.
    VaR_index = int(len(sorted_data) * 0.05)
    VaR = -sorted_data[VaR_index]

    # Calculate the relative difference from the mean expected.
    VaR_diff = VaR + mu

    # Calculate the ES
    ES = -sorted_data[sorted_data <= -VaR].mean()

    # Calculate the relative difference from the mean expected.
    ES_diff = ES + mu

    return pd.DataFrame({"VaR Absolute": [VaR],
                         "VaR Diff from Mean": [VaR_diff],
                         "ES Absolute": [ES],
                         "ES Diff from Mean": [ES_diff]})
