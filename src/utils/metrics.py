from sklearn.metrics import confusion_matrix,mean_squared_error,mean_absolute_error,r2_score
from scipy.stats import pearsonr
import numpy as np

def compute_reg_metrics(ypred, ytrue):
    mae = mean_absolute_error(ytrue, ypred)
    rmse = mean_squared_error(y_true=ytrue, y_pred=ypred, squared=False)
    r2 = r2_score(y_true=ytrue, y_pred=ypred)
    pcc, _ = pearsonr(ytrue, ypred)

    return mae, rmse, r2, pcc

def count_positive_products(preds, labels):

    if not (isinstance(preds, np.ndarray) and isinstance(labels, np.ndarray)):
        raise TypeError("Not NumPy ndarray")

    if preds.shape != labels.shape:
        raise ValueError("Shape must be the same")

    product = preds * labels
    return np.count_nonzero(product > 0)