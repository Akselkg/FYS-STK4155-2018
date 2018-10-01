import numpy as np
import numpy.random as rng
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


np.set_printoptions(linewidth=1000, precision=5, suppress=True, threshold=1000000)
rng.seed(1)

# create tuple representing design matrix, in order: 1, x, y, x**2, x*y, y**2, x**3, x**2*y, y**2*x, y**3, x**4, ...)
# n = max(i+j) in (x^i*y^j)
def sol_tup(x, y, n):
    tup = (np.ones_like(x),)
    for i in range(1, n+1):
        for j in range(i + 1):
            tup += (x**(i-j) * y**j,)
    return tup


def k_cross_Franke(n=100, k=10, eps=0):
    max_order = 5
    x = rng.uniform(size=n); y = rng.uniform(size=n); err = eps * rng.normal(size=n)

    sub_arrays_x = np.array_split(x, k); sub_arrays_y = np.array_split(y, k); sub_arrays_err = np.array_split(err, k)
    mse = np.zeros((k, max_order))
    R2 = np.zeros((k, max_order))


    for k_ in range(k):
        x_train = np.concatenate(sub_arrays_x[:k_] + sub_arrays_x[k_+1:])
        x_valid = sub_arrays_x[k_]
        y_train = np.concatenate(sub_arrays_y[:k_] + sub_arrays_y[k_+1:])
        y_valid = sub_arrays_y[k_]
        err_train = np.concatenate(sub_arrays_err[:k_] + sub_arrays_err[k_+1:])
        err_valid = sub_arrays_err[k_]
        z_train = FrankeFunction(x_train, y_train) + err_train

        x_valid_, y_valid_ = np.meshgrid(x_valid, y_valid)
        z_valid = FrankeFunction(x_valid_, y_valid_) + err_valid



        for order in range(1, max_order+1):
            xb = np.column_stack(sol_tup(x_train, y_train, order))
            beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(z_train)
            zhat = np.array(sol_tup(x_valid_, y_valid_, order)).T.dot(beta).T
            mse[k_, order-1] = np.mean((z_valid - zhat)**2)
            R2[k_, order-1] = 1 - np.mean((z_valid - zhat)**2) / np.mean((z_valid - np.mean(z_valid))**2)

    mse = np.mean(mse, axis=0)
    R2 = np.mean(R2, axis=0)

    return mse, R2

#  bad
def k_cross_data(z, n=10000, k=10):
    max_order = 5

    xn = z.shape[0]
    yn = z.shape[1]
    x = rng.choice(range(xn), n); y = rng.choice(range(yn), n)

    sub_arrays_x = np.array_split(x, k); sub_arrays_y = np.array_split(y, k);
    mse = np.zeros((k, max_order))
    R2 = np.zeros((k, max_order))

    for k_ in range(k):
        x_train = np.concatenate(sub_arrays_x[:k_] + sub_arrays_x[k_+1:])
        print("x",x_train)
        x_valid = sub_arrays_x[k_]
        y_train = np.concatenate(sub_arrays_y[:k_] + sub_arrays_y[k_+1:])
        print("y",y_train)
        y_valid = sub_arrays_y[k_]
        z_train = z[x_train, y_train]
        print("z",z_train)
        z_valid = z[x_valid, y_valid]

        for order in range(1, max_order+1):
            xb = np.column_stack(sol_tup(x_train, y_train, order))
            beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(z_train)
            zhat = np.column_stack(sol_tup(x_valid, y_valid, order)).dot(beta)
            mse[k_, order-1] = np.mean((z_valid - zhat)**2)
            R2[k_, order-1] = 1 - np.mean((z_valid - zhat)**2) / np.mean((z_valid - np.mean(z_valid))**2)

        print(xb)
        print(beta)
        print(zhat)
    mse = np.mean(mse, axis=0)
    R2 = np.mean(R2, axis=0)

    return mse, R2

def k_cross_Franke_ridge(n=1000, k=10, eps=0):
    m = 50
    l = np.exp(np.linspace(-20, 3, m))
    order = 5
    x = rng.uniform(size=n); y = rng.uniform(size=n); err = eps * rng.normal(size=n)

    sub_arrays_x = np.array_split(x, k); sub_arrays_y = np.array_split(y, k); sub_arrays_err = np.array_split(err, k)
    mse = np.zeros((k, m))
    R2 = np.zeros((k, m))


    for k_ in range(k):
        x_train = np.concatenate(sub_arrays_x[:k_] + sub_arrays_x[k_+1:])
        x_valid = sub_arrays_x[k_]
        y_train = np.concatenate(sub_arrays_y[:k_] + sub_arrays_y[k_+1:])
        y_valid = sub_arrays_y[k_]
        err_train = np.concatenate(sub_arrays_err[:k_] + sub_arrays_err[k_+1:])
        err_valid = sub_arrays_err[k_]
        z_train = FrankeFunction(x_train, y_train) + err_train

        x_valid_, y_valid_ = np.meshgrid(x_valid, y_valid)
        z_valid = FrankeFunction(x_valid_, y_valid_) + err_valid

        for i_l, l_ in enumerate(l):
            xb = np.column_stack(sol_tup(x_train, y_train, order))
            l_eye = l_ * np.eye(xb.shape[1])
            beta = np.linalg.inv(xb.T.dot(xb) + l_eye).dot(xb.T).dot(z_train)
            zhat = np.array(sol_tup(x_valid_, y_valid_, order)).T.dot(beta).T
            mse[k_, i_l] = np.mean((z_valid - zhat)**2)
            R2[k_, i_l] = 1 - np.mean((z_valid - zhat)**2) / np.mean((z_valid - np.mean(z_valid))**2)

    mse = np.mean(mse, axis=0)
    R2 = np.mean(R2, axis=0)

    return mse, R2, l


def franke_lasso(n=10000, eps=0.0):
    max_order = 5
    x = rng.uniform(size=n); y = rng.uniform(size=n); err = eps * rng.normal(size=n)

    x_train = x[:int(n/2)]; y_train = y[:int(n/2)]; err_train = err[:int(n/2)]
    x_valid = x[int(n/2):]; y_valid = y[int(n/2):]; err_valid = err[:int(n/2)];

    z_train = FrankeFunction(x_train, y_train) + err_train
    z_valid = FrankeFunction(x_valid, y_valid) + err_valid
    xb = np.column_stack(sol_tup(x_train, y_train, max_order))

    lasso=linear_model.LassoCV(max_iter=100000, cv=5)
    lasso.fit(xb, z_train)
    predl=lasso.predict(np.column_stack(sol_tup(x_valid, y_valid, max_order)))

    return lasso.coef_, mean_squared_error(z_valid, predl), r2_score(z_valid, predl)
