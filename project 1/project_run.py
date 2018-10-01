import numpy as np
import numpy.random as rng
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from project_functions import *
from imageio import imread

n = 200
order = 4
eps = 0
x = rng.uniform(size=n)
y = rng.uniform(size=n)
err = eps * rng.normal(size=n)

z = FrankeFunction(x, y) + err
xb = np.column_stack(sol_tup(x,y, order))
beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(z)

# Make xy grid
x = np.linspace(0, 1, 40)
y = np.linspace(0, 1, 40)
x, y = np.meshgrid(x,y)
zhat = np.array(sol_tup(x,y,order)).T.dot(beta).T
z = FrankeFunction(x, y)

# statistics
mse = np.mean((z-zhat)**2)
R2 = 1 - np.mean((z-zhat)**2) / np.mean((z-np.mean(z))**2)
#print("mse: ", mse, " R2: ", R2, " order: ", order)

# Plot the surface.
def plot_franke(x, y, z, cmap=cm.coolwarm, name="plot3d.png"):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cmap, linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('3dplt.png')

def ex_k_cross_Franke():
    eps = 0.1
    mse, R2 = k_cross_Franke(eps=eps)
    print("10-fold cross validation, eps = %.2f: " % eps)
    for mse_, R2_, order_ in zip(mse, R2, range(1, 6)):

        print("order: %d" % order_, " MSE: %5f" % mse_, " R2: %5f" % R2_)


def ex_k_cross_Franke_ridge(n=200):

    plt.figure()
    eps = 0
    mse, R2, l = k_cross_Franke_ridge(n, eps=eps)
    print(l[np.argmin(mse)], np.min(mse), R2[np.argmin(mse)], eps)
    plt.subplot(221)

    plt.semilogx(l, mse, label="eps = %.2f" % eps)
    plt.semilogx(l[np.argmin(mse)], np.min(mse), 'o')
    plt.legend()
    plt.xlabel(r"$\lambda$")

    plt.ylabel("MSE")

    eps = 0.1
    mse, R2, l = k_cross_Franke_ridge(n, eps=eps)
    print(l[np.argmin(mse)], np.min(mse), R2[np.argmin(mse)], eps)
    plt.subplot(222)

    plt.semilogx(l, mse, label="eps = %.2f" % eps)
    plt.semilogx(l[np.argmin(mse)], np.min(mse), 'o')
    plt.legend()
    plt.xlabel(r"$\lambda$")

    plt.ylabel("MSE")

    eps = 0.2
    mse, R2, l = k_cross_Franke_ridge(n, eps=eps)
    print(l[np.argmin(mse)], np.min(mse), R2[np.argmin(mse)], eps)

    plt.subplot(223)

    plt.semilogx(l, mse, label="eps = %.2f" % eps)
    plt.semilogx(l[np.argmin(mse)], np.min(mse), 'o')
    plt.legend()
    plt.xlabel(r"$\lambda$")

    plt.ylabel("MSE")

    eps = 0.5
    mse, R2, l = k_cross_Franke_ridge(n, eps=eps)
    print(l[np.argmin(mse)], np.min(mse), R2[np.argmin(mse)], eps)

    plt.subplot(224)
    plt.semilogx(l, mse, label="eps = %.2f" % eps)
    plt.semilogx(l[np.argmin(mse)], np.min(mse), 'o')
    plt.legend()
    plt.xlabel(r"$\lambda$")

    plt.ylabel("MSE")

    plt.savefig('k_cross_ridge.png')

def print_lasso_franke(n = 10000):
    for eps in [0, 0.1, 0.2, 0.5]:
        beta, mse, R2 = franke_lasso(n, eps=eps)
        print(" MSE: %5f" % mse, " R2: %5f" % R2 , " eps: %2f" % eps)

"""
terrain1 = imread('SRTM_data_Norway_1.tif')
print(terrain1)


plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

mse, R2 = k_cross_data(terrain1, n=20)
print(mse, R2)
"""
ex_k_cross_Franke()
ex_k_cross_Franke_ridge()
print_lasso_franke()
plot_franke(x, y, z, cmap=cm.coolwarm, name="plot_real.png")
plot_franke(x, y, zhat, cmap=cm.plasma, name="plot_reg.png")
plt.show()
