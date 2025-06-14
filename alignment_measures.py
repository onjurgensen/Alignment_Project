# Import necessary libraries
import numpy as np
import pandas as pd
import torch

# regression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torchmetrics.functional import pearson_corrcoef

# linear shape metric
from netrep.metrics import LinearMetric
import similarity




######################### Regression ##########################

def pls_regression(X_train, Y_train, X_test, Y_test, n_components=1, multioutput= "raw_values", standardize=False, permute = False):
    """
    Partial Least Squares Regression (PLS) adapted from sklearn.
    Returns the R^2 score for PLS regression between two matrices.

    - X_train and X_test are numpy arrays of shape (n_samples, n_features)
    - Y_train and Y_test are numpy arrays of shape (n_samples, n_targets)
    - For standard linear predictivity, X should be DNNS features and Y should be brain data.
    - Move data to CPU and convert to numpy before calling this function.
    """
    if standardize:
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_Y.transform(X_test)

    predictor = PLSRegression(n_components=n_components)
    predictor.fit(X_train, Y_train)
    Y_pred = predictor.predict(X_test)

    if standardize:
        Y_pred = scaler_Y.inverse_transform(Y_pred)
        Y_test = scaler_Y.inverse_transform(Y_test)
    
    r2_scores = r2_score(y_pred=Y_pred, y_true=Y_test, multioutput=multioutput)
    pearson_scores = pearson_corrcoef(torch.tensor(Y_pred), torch.tensor(Y_test)).numpy()

    if permute:
        Y_test_permuted = Y_test[np.random.permutation(Y_test.shape[0])]
        r2_scores_permuted = r2_score(y_pred=Y_pred, y_true=Y_test_permuted, multioutput=multioutput)
        pearson_scores_permuted = pearson_corrcoef(torch.tensor(Y_pred), torch.tensor(Y_test_permuted)).numpy()
        
        return r2_scores, pearson_scores, r2_scores_permuted, pearson_scores_permuted
    
    return r2_scores, pearson_scores




def ridge_regression(X_train, X_test, Y_train, Y_test, alpha=1.0, multioutput= "raw_values", standardize=False,
                     return_predictions = False, permute = False):
    
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

    r2_scores = r2_score(y_pred=Y_pred, y_true=Y_test, multioutput=multioutput)
    pearson_scores = pearson_corrcoef(torch.tensor(Y_pred), torch.tensor(Y_test)).numpy()

    if permute & return_predictions:
        # Permutation test
        Y_test_permuted = Y_test[np.random.permutation(Y_test.shape[0])]
        r2_scores_permuted = r2_score(y_pred=Y_pred, y_true=Y_test_permuted, multioutput=multioutput)
        pearson_scores_permuted = pearson_corrcoef(torch.tensor(Y_pred), torch.tensor(Y_test_permuted)).numpy()

        return r2_scores, pearson_scores, r2_scores_permuted, pearson_scores_permuted, Y_pred
    
    if return_predictions == False:
        return r2_scores, pearson_scores
    else:
        return r2_scores, pearson_scores, Y_pred


def ridge_regression_cv(X_train, Y_train, X_test, Y_test, alphas=np.logspace(-8, 8, 17), standardize=False, 
                        multioutput = "raw_values", return_predictions = False, permute = False):
    """
    return predictions was added to speed uo versa calculations
    """

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

    r2_scores = r2_score(y_pred=Y_pred, y_true=Y_test, multioutput=multioutput)
    pearson_scores = pearson_corrcoef(torch.tensor(Y_pred), torch.tensor(Y_test)).numpy()



    if permute & return_predictions:
        # Permutation test
        Y_test_permuted = Y_test[np.random.permutation(Y_test.shape[0])]
        r2_scores_permuted = r2_score(y_pred=Y_pred, y_true=Y_test_permuted, multioutput=multioutput)
        pearson_scores_permuted = pearson_corrcoef(torch.tensor(Y_pred), torch.tensor(Y_test_permuted)).numpy()

        return r2_scores, pearson_scores, r2_scores_permuted, pearson_scores_permuted, Y_pred
    
    if return_predictions == False:
        return r2_scores, pearson_scores
    else:
        return r2_scores, pearson_scores, Y_pred

    



########################## RSA ##########################


def rsa(X, Y, metric='correlation', method='spearman'):

    """
    use 'correlation' or 'squared_euclidean' for metric
    
    """
    if metric != "cosine":
        rsa = similarity.make(f"measure/rsatoolbox/rsa-rdm={metric}-compare={method}")
    else:
        rsa = similarity.make(f"measure/thingsvision/rsa-rdm=cosine-compare={method}")


    return(rsa(X, Y))


def versa(X_train, Y_train, X_test, Y_test, metrics= ["correlation"], methods = ["spearman"], alphas=np.logspace(-8, 8, 17), standardize=False, time_series=False):
    """
    Perform ridge regression with optional standardization and dimensionality reduction,
    followed by representational similarity analysis (RSA).

    Parameters:
    - X_train, X_test: Feature matrices (numpy arrays).
    - Y_train, Y_test: Target matrices (numpy arrays).
    - metrics (tuple): Metrics for RSA (default: ("correlation")).
    - methods (tuple): Methods for RSA (default: ("spearman")).
    - alphas: Regularization strengths for RidgeCV (default: np.logspace(-8, 8, 17)).
    - standardize: Whether to standardize features and targets (default: False).
    - dim_reduction: Dimensionality reduction method (e.g., "srp" for Sparse Random Projection).

    Returns:
    - RSA result comparing predicted and actual target data.
    """
    if time_series is False:

        if standardize:
            # Standardize features and targets
            scaler_X = StandardScaler()
            scaler_Y = StandardScaler()

            X_train = scaler_X.fit_transform(X_train)
            X_test = scaler_X.transform(X_test)
            Y_train = scaler_Y.fit_transform(Y_train)
            Y_test = scaler_Y.transform(Y_test)

        # Ridge regression with cross-validation
        predictor = RidgeCV(alphas=alphas)
        predictor.fit(X_train, Y_train)
        Y_pred = predictor.predict(X_test)

        # Inverse transform if standardization was applied
        if standardize:
            Y_pred = scaler_Y.inverse_transform(Y_pred)
            Y_test = scaler_Y.inverse_transform(Y_test)

        # Perform RSA
        measure_scores = []
        for metric in metrics:
            for method in methods:
                if metric != "cosine":
                    rsa = similarity.make(f"measure/rsatoolbox/rsa-rdm={metric}-compare={method}")
                else:
                    rsa = similarity.make(f"measure/thingsvision/rsa-rdm=cosine-compare={method}")
                # Compute RSA
                rsa_score = rsa(Y_pred, Y_test)
                
        return measure_scores

    if time_series is True:

        Y_train_flat = Y_train.reshape(Y_train.shape[0], -1)
        Y_test_flat = Y_test.reshape(Y_test.shape[0], -1)

        predictor = RidgeCV(alphas=alphas)
        predictor.fit(X_train, Y_train_flat)
        Y_pred = predictor.predict(X_test)

        # Reshape Y_pred to match the shape of Y_test
        Y_pred = Y_pred.reshape(Y_test.shape)

        # Perform RSA
        measure_scores = []
        # Compute RSA for each setting
        for metric in metrics:
            for method in methods:
                if metric != "cosine":
                    rsa = similarity.make(f"measure/rsatoolbox/rsa-rdm={metric}-compare={method}")
                else:
                    rsa = similarity.make(f"measure/thingsvision/rsa-rdm=cosine-compare={method}")
                # Compute RSA for each timepoint
                timepoint_scores = []
                for i in range(Y_pred.shape[2]):
                    rsa_score = rsa(Y_pred[:, :, i], Y_test[:, :, i])
                    timepoint_scores.append(rsa_score)
                measure_scores.append(timepoint_scores)

        return measure_scores

########################## Linear Shape Metric ##########################


def linear_shape_metric(X_train, Y_train, X_test, Y_test, alpha = 1, score_method = "angular", zero_pad = True, permute = False):
    """
    Compute the linear shape metric between two sets of data.
    
    Parameters:
    - X_train: Training data (numpy array).
    - Y_train: Training labels (numpy array).
    - X_test: Test data (numpy array).
    - Y_test: Test labels (numpy array).
    - alpha: Regularization parameter for the linear shape metric.
      - alpha = 0.0: CCA metric (no regularization).
      - alpha = 1.0: Procrustes metric (fully regularized metric)
    
    Returns:
    - Linear shape metric value.
    """
    
    # Compute the linear shape metric
    metric = LinearMetric(alpha=alpha, center_columns=True, score_method = score_method, zero_pad=zero_pad)

    # Fit the metric to the training data
    metric.fit(X_train, Y_train)
    # get score on training data
    score_train = metric.score(X_train, Y_train)
    # get score on test data
    score = metric.score(X_test, Y_test)

    if permute == True:
        # Permutation test
        Y_test_permuted = Y_test[np.random.permutation(Y_test.shape[0])]
        score_permuted = metric.score(X_test, Y_test_permuted)
        return score_train, score, score_permuted
    if permute == False:
        return score_train, score


def linear_shape_metric_cv(X_train, Y_train, X_test, Y_test, alphas=np.linspace(0, 1, 11), score_method="angular", cv_folds=5):
    """
    Compute the linear shape metric with cross-validation for the alpha parameter.
    
    Parameters:
    - X_train: Training data (numpy array).
    - Y_train: Training labels (numpy array).
    - X_test: Test data (numpy array).
    - Y_test: Test labels (numpy array).
    - alphas: List or array of alpha values to cross-validate.
    - score_method: Scoring method for the linear shape metric.
    - cv_folds: Number of folds for cross-validation.
    
    Returns:
    - Best alpha value based on cross-validation.
    - Linear shape metric value using the best alpha.
    """

    # If multiple alphas provided, perform cross-validation to find the best alpha
    if len(alphas) > 1:

        best_alpha = None
        best_score = np.inf
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for alpha in alphas:
            cv_scores = []
            for train_idx, val_idx in kf.split(X_train):
                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                Y_train_fold, Y_val_fold = Y_train[train_idx], Y_train[val_idx]

                # Compute the linear shape metric
                metric = LinearMetric(alpha=alpha, center_columns=True, score_method=score_method)
                metric.fit(X_train_fold, Y_train_fold)
                score = metric.score(X_val_fold, Y_val_fold)
                cv_scores.append(score)

            # Average score for this alpha
            mean_cv_score = np.mean(cv_scores)
            if mean_cv_score < best_score:  # Minimize the score
                best_score = mean_cv_score
                best_alpha = alpha

    # Compute the final score on the test set using the best alpha
    metric = LinearMetric(alpha=best_alpha, center_columns=True, score_method=score_method)
    metric.fit(X_train, Y_train, score_method)
    final_score = metric.score(X_test, Y_test)

    return best_alpha, final_score



def cka(X, Y, output = "score"):
    """
    Compute the linear Centered Kernel Alignment (CKA) between two matrices.

    output: str
        - "score", distance=squared_euclidean", "distance=euclidean", or "distance=angular"

    Returns:
    - CKA value.
    """
    # Compute the CKA value
    cka_score = similarity.make(f"measure/netrep/cka-kernel=linear-hsic=gretton-{output}")
    
    return cka_score(X, Y)

def cka_debiased(X, Y):
    """
    Compute the linear Centered Kernel Alignment (CKA) between two matrices.

    Returns:
    - CKA value.
    """
    # Compute the CKA value
    cka_score = similarity.make("correcting_cka_alignment/cka_debiased")
    
    return cka_score(X, Y)