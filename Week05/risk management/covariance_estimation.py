import numpy as np
import pandas as pd

'''
Function to calculate the covariance matrix for the dataframe that does not have the entire data.
When skipRow is true, use all the rows which have values. When it's false, use pairwise.
func can be np.cov (covariance) and np.corrcoef (correlation)
'''


def missing_cov_corr(df, skipRow=True, func=np.cov):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    m, n = df.shape
    missing_rows = df.isnull().any(axis=1).sum()

    # If there is no missing rows, simply calculate the covariance matrix.
    if not missing_rows:
        cov = func(df.T)
        print(cov)
    else:
        # skipRow is True, apply the method on the rows that have all the data.
        if skipRow:
            # Drop the rows that is not of whole data
            df = df.dropna(axis=0, how='any')
            cov = func(df.T)
        # skipRow is True, apply the pairwise method.
        else:
            out = np.empty((n, n))
            for i in range(n):
                for j in range(i + 1):
                    # Select only rows without missing values in either column i or j
                    valid_rows = df.iloc[:, [i, j]].dropna().index

                    if not valid_rows.empty:
                        cov_ij = func(df.iloc[valid_rows, [i, j]], rowvar=False)[0, 1]
                        out[i, j] = cov_ij
                        out[j, i] = cov_ij
                        cov = out
    return cov


'''
Function that calculates the EW covariance and correlation.
Func take the parameter of 'cov' and 'corr'
'''


def ew_cov_corr(df, lmbd, func='cov'):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if func not in ['cov', 'corr']:
        raise ValueError(f'The func parameter must be "cov" or "corr", got {func} instead.')

    # Center the data - to calculate the covariance matrix.
    df -= df.mean(axis=0)

    m, n = df.shape
    wts = np.empty(m)

    # Setting weights for prior observation
    for i in range(m):
        wts[i] = (1 - lmbd) * lmbd ** (m - i - 1)

    # Normalizing the weights
    wts /= np.sum(wts)
    wts = wts.reshape(-1, 1)
    if func == 'cov':
        res = (wts * df).T @ df

    elif func == 'corr':
        res = (wts * df).T @ df
        # Calculate the standard deviations (square root of variances along the diagonal)
        std_devs = np.sqrt(np.diag(res))

        # Convert the covariance matrix to a correlation matrix
        res /= np.outer(std_devs, std_devs)

    return res
