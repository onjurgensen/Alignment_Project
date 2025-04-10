# Import necessary libraries
import numpy as np
import pandas as pd

# regression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection

# linear shape metric
from netrep.metrics import LinearMetric
import similarity

# RSA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from scipy.stats import spearmanr


######################### Regression ##########################

def pls_regression(X_train, X_test, Y_train, Y_test, n_components=1, standardize=False):
    """
    Partial Least Squares Regression (PLS) adapted from sklearn.
    Returns the R^2 score for PLS regression between two matrices.

    - X_train and X_test are numpy arrays of shape (n_samples, n_features)
    - Y_train and Y_test are numpy arrays of shape (n_samples, n_targets)
    - For standard linear predictivity, X should be DNNS features and Y should be brain data.
    - Move data to CPU and convert to numpy before calling this function.
    """
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    predictor = PLSRegression(n_components=n_components)
    predictor.fit(X_train, Y_train)
    Y_pred = predictor.predict(X_test)

    return r2_score(y_pred=Y_pred, y_true=Y_test, multioutput="raw_values")



def ridge_regression(X_train, X_test, Y_train, Y_test, alpha=1.0, standardize=False):
    
    if standardize:
        # Standardize the features
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()

        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        Y_train = scaler_Y.fit_transform(Y_train)
        Y_test = scaler_Y.transform(Y_test)
        
    predictor = Ridge(alpha=alpha)
    predictor.fit(X_train, Y_train)
    Y_pred = predictor.predict(X_test)

    # If standardization was applied, inverse transform Y_test and Y_pred for proper comparison
    if standardize:
        Y_pred = scaler_Y.inverse_transform(Y_pred)
        Y_test = scaler_Y.inverse_transform(Y_test)

    return r2_score(y_pred=Y_pred, y_true=Y_test, multioutput="raw_values")


def ridge_regression_cv(X_train, Y_train, X_test, Y_test, alphas=np.logspace(-8, 8, 17), standardize=False):

    if standardize:
        # Standardize the features
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()

        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        Y_train = scaler_Y.fit_transform(Y_train)
        Y_test = scaler_Y.transform(Y_test)
        
    predictor = RidgeCV(alphas=alphas)
    predictor.fit(X_train, Y_train)
    Y_pred = predictor.predict(X_test)

    # If standardization was applied, inverse transform Y_test and Y_pred for proper comparison
    if standardize:
        Y_pred = scaler_Y.inverse_transform(Y_pred)
        Y_test = scaler_Y.inverse_transform(Y_test)

    return r2_score(y_pred=Y_pred, y_true=Y_test, multioutput="raw_values")



########################## RSA ##########################


def rsa(X, Y, metric='correlation', method='spearman'):
    rsa = similarity.make("measure/rsatoolbox/rsa-rdm={metric}-compare={method}")

    return(rsa(X, Y))

def versa(X_train, Y_train, X_test, Y_test, metric="correlation", method="spearman", alphas = np.logspace(-8, 8, 17), standardize=False, dim_reduction=None):

    if standardize:
        # Standardize the features
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()

        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        Y_train = scaler_Y.fit_transform(Y_train)
        Y_test = scaler_Y.transform(Y_test)

    if dim_reduction == "srp":
        # Apply Sparse Random Projection
        srp = SparseRandomProjection(alpha =0.1)
        X_train = srp.fit_transform(X_train)
        X_test = srp.transform(X_test)
        
    predictor = RidgeCV(alphas=alphas)
    predictor.fit(X_train, Y_train)
    Y_pred = predictor.predict(X_test)

    # If standardization was applied, inverse transform Y_test and Y_pred for proper comparison
    if standardize:
        Y_pred = scaler_Y.inverse_transform(Y_pred)
        Y_test = scaler_Y.inverse_transform(Y_test)

    rsa = similarity.make("measure/rsatoolbox/versa-rdm={metric}-compare={method}")

    return rsa(Y_pred, Y_test)

