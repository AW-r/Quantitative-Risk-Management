import pandas as pd
from simulation import fit_normal, fit_general_t
from scipy.stats import norm, t

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
VaR for t Distribution simulation
'''


def var_simulation(data, alpha=0.05, size=10000):
    # Fit the data with t distribution.
    mu, sigma, nu = fit_general_t(data)

    # Generate given size random numbers from a t-distribution
    random_numbers = t.rvs(df=nu, loc=mu, scale=sigma, size=size)

    return var_t(random_numbers, alpha)


'''
VaR for Normal distribution with an EW variance
'''


def var_ew(data, lmbd, alpha=0.05):
    # Calculate the mean.
    mu = data.mean()

    # Calculate the variance with an EW variance.
    ew_sigma2 = ew_cov_corr(data, lmbd, 'cov')

    # Calculate the VaR
    VaR = -norm.ppf(alpha, mu, np.sqrt(ew_sigma2))[0, 0]

    # Calculate the relative difference from the mean expected.
    VaR_diff = VaR + mu
    VaR_diff = VaR_diff[0]

    return pd.DataFrame({"VaR Absolute": [VaR],
                         "VaR Diff from Mean": [VaR_diff]})


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
