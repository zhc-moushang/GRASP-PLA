import numpy as np
import sklearn.metrics as m
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, cohen_kappa_score

from numba import njit

@njit
def c_index(y_true, y_pred):
    summ = 0
    pair = 0

    for i in range(1, len(y_true)):
        for j in range(0, i):
            pair += 1
            if y_true[i] > y_true[j]:
                summ += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            elif y_true[i] < y_true[j]:
                summ += 1 * (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            else:
                pair -= 1

    if pair is not 0:
        return summ / pair
    else:
        return 0


def RMSE(y_true, y_pred):
    return np.sqrt(m.mean_squared_error(y_true, y_pred))


def MAE(y_true, y_pred):
    return m.mean_absolute_error(y_true, y_pred)


def CORR(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def SD(y_true, y_pred):
    from sklearn.linear_model import LinearRegression
    y_pred = y_pred.reshape((-1,1))
    lr = LinearRegression().fit(y_pred,y_true)
    y_ = lr.predict(y_pred)
    return np.sqrt(np.square(y_true - y_).sum() / (len(y_pred) - 1))

def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci
def rm2(y,f):
    r2 = r_squared_error(y, f)
    r02 = squared_error_zero(y, f)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))
def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)
def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))

def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse

def calculate_auc(y_true, y_pred):
    """
    计算AUC值，使用真实值的平均值作为阈值。

    参数:
    y_true: array-like, 真实标签 (0和1)
    y_pred: array-like, 模型预测值（可以是回归输出）

    返回:
    auc_score: float, 计算得到的AUC值
    """
    # 计算真实值的平均值
    threshold = np.mean(y_true)

    # 将预测值转换为二进制标签

    y_true_binary = (y_true > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)

    # 计算AUC
    auc_score = roc_auc_score(y_true_binary, y_pred_binary)

    return auc_score