# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
print('Matplotlib version: {}'.format(mpl.__version__))
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
cmap = mpl.cm.Set1

from matplotlib.patches import Circle, Rectangle

cmap = mpl.cm.spectral
myred = '#c1151a'


# Shoelace formula
def PolygonArea(corners):
    n = len(corners)  # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def rungekutta4(fun_yprime, t, y0, h):
    y = [y0]
    for i in range(len(t) - 1):
        K1 = h * fun_yprime(y[i], t[i])
        K2 = h * fun_yprime(y[i] + K1 / 2, t[i] + h / 2)
        K3 = h * fun_yprime(y[i] + K2 / 2, t[i] + h / 2)
        K4 = h * fun_yprime(y[i] + K3, t[i] + h)
        y.append(y[i] + (K1 + 2 * K2 + 2 * K3 + K4) / 6)
    return y


# hamilton function H = sin(x) + sin(p)
def fun(y, t):
    return np.array([y[1] + t * 2, -0.5 * np.cos(y[0]) + t ** 0.5])


radius = 5

# inside particles
N = 30
r_vec = 2 * (np.random.rand(1, N) - 0.5)
phi_vec = (2 * np.random.rand(1, N) - 1) * 2 * np.pi
# r_vec = np.linspace(0, 1, N).reshape(1, N)
# phi_vec = np.linspace(0, 2 * np.pi, N).reshape(1, N)
# q = radius / 2 * np.cos(phi_vec)
# p = radius / 2 * np.sin(phi_vec)
q = radius * r_vec * np.cos(phi_vec)
p = radius * r_vec * np.sin(phi_vec)

y0 = np.vstack((q, p))

# outside particles
N_outside = 500
phi_vec_outside = np.linspace(0, 2 * np.pi, N_outside).reshape(1, N_outside)
q_outside = radius * np.cos(phi_vec_outside)
p_outside = radius * np.sin(phi_vec_outside)
y0_outside = np.vstack((q_outside, p_outside))

if len(sys.argv) > 1:
    stepspertime = int(sys.argv[1])
    print('steps per time {}'.format(stepspertime))
else:
    stepspertime = 100
time = 5
t, h = np.linspace(0, time, stepspertime * time, retstep=True)

y = rungekutta4(fun, t, y0, h)
y_outside = rungekutta4(fun, t, y0_outside, h)
y_0 = [x[0, :] for x in y]
y_1 = [x[1, :] for x in y]
y_0_outside = [x[0, :] for x in y_outside]
y_1_outside = [x[1, :] for x in y_outside]

if __name__ == '__main__':
    fig = plt.figure('Liouville\'s theorem', facecolor='1', figsize=(12, 6))
    ax = fig.add_subplot(111)
    plt.plot(y_0, y_1, '-k', alpha=0.25)
    for i in range(N):
        plt.plot(y[0][0, i], y[0][1, i], 'o', color=myred, markeredgewidth=0.0)
        plt.plot(y[-1][0, i], y[-1][1, i], 'o', color=myred, markeredgewidth=0.0)
    plt.plot(y_0_outside[0], y_1_outside[0], '-k', linewidth=1.5)
    plt.plot(y_0_outside[-1], y_1_outside[-1], '-k', linewidth=1.5)
    ax.grid(linestyle='--', alpha=0.85)
    # plt.gca().add_artist(Rectangle((-9, 10.5), 28, 4, facecolor='white', edgecolor="black", zorder=3))
    ax.set_xlabel('$q$')
    ax.set_ylabel('$p$')
    # ax.set_title('Liouville\'s theorem')
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    corners_0 = list(zip(y_0_outside[0], y_1_outside[0]))
    area_0 = PolygonArea(corners_0)
    corners_t = list(zip(y_0_outside[-1], y_1_outside[-1]))
    area_t = PolygonArea(corners_t)
    ax.annotate('$G_0$', xy=(4, -2.5), xytext=(13, -2.5), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=26)
    ax.annotate('$G_t$', xy=(40, 6.4), xytext=(50, 3.5), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=26)

    props = dict(boxstyle='round', facecolor='white')
    # plt.annotate('$\\frac{p^2}{2} + 2 t p  + \\frac{\sin{q}}{2} - q \sqrt{t}$', xy=(.085, .88), xycoords='figure fraction', ha='left', fontsize=18, zorder=10)
    # plt.annotate('area $G_0$: {:.3f}'.format(area_0), xy=(.085, .83), xycoords='figure fraction', ha='left', fontsize=16, zorder=10)
    # plt.annotate('area $G_t$: {:.3f}'.format(area_t), xy=(.085, .78), xycoords='figure fraction', ha='left', fontsize=16, zorder=10)
    plt.gca().text(0.025, 0.95, '$H(q,p,t) = \\frac{p^2}{2} + 2 t p  + \\frac{\sin{q}}{2} - q \sqrt{t}$ \n\n' + 'area $G_0$: {:.3f} \n'.format(area_0) + 'area $G_t$: {:.3f}'.format(area_t), transform=plt.gca().transAxes, fontsize=18, verticalalignment='top', bbox=props)



    plt.tight_layout()
    plt.savefig('../images/A-liouville-theorem-constant-volume.pdf')
