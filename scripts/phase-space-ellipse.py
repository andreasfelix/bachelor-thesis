# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl


print('Matplotlib version: {}'.format(mpl.__version__))
SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize



# from matplotlib import rc
# # for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)

import sys
from os.path import expanduser
sys.path.append(expanduser("~")+"/git")


from element.pyfiles.compute.transfermatrix_element_slicing import returnlatticedata
from element.pyfiles.compute.tracking import returntrackingdata


cmap = mpl.cm.spectral


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


if __name__ == '__main__':
    # Klasse für Attribute
    class AttrClass:
        pass


    tracksett = AttrClass
    tracksett.rounds = 1
    tracksett.N = 40
    Emittanz = 5.3e-9  # m rad
    width = Emittanz ** 0.5 * 1000  # mm oder mrad

    # outside particles
    N_outside = tracksett.N
    r_vec_outside = width
    phi_vec_outside = np.linspace(0, 2 * np.pi, N_outside).reshape(1, N_outside)
    x_outside = r_vec_outside * np.cos(phi_vec_outside)
    dxds_outside = r_vec_outside * np.sin(phi_vec_outside)
    y_outside = np.zeros((1, N_outside))
    dyds_outside = np.zeros((1, N_outside))
    delta_outside = 0 * np.ones((1, N_outside))
    tracksett.X0 = np.vstack((x_outside, dxds_outside, y_outside, dyds_outside, delta_outside))

    zoom = 1.5


    def plotphasespace(title):
        currentmagnettype = latticedata.ElementType[0]
        ax.grid(linestyle='--', linewidth=.7, alpha=0.9)
        corners = list(zip(trackingdata.xtrack[0, :], trackingdata.dxdstrack[0, :]))
        area = PolygonArea(corners)
        ax.set_ylim(-zoom * width, zoom * width)
        ax.set_xlim(-zoom * width, zoom * width)
        plt.title(title)
        plt.plot(trackingdata.xtrack, trackingdata.dxdstrack, '-k', alpha=0.5)
        plt.xlabel('$u$ / mm')
        plt.ylabel('$u\'$ / mrad')
        plt.xticks([-0.1, 0 , 0.1])
        plt.yticks(plt.gca().get_xticks())
        for N in range(tracksett.N):
            plt.plot(trackingdata.xtrack[0, N], trackingdata.dxdstrack[0, N], 'o', color=cmap(N / tracksett.N), markeredgewidth=0.0)
            plt.plot(trackingdata.xtrack[-1, N], trackingdata.dxdstrack[-1, N], 'o', color=cmap(N / tracksett.N), markeredgewidth=0.0)


    fig = plt.figure('Phase space ellipse transformation', facecolor='1', figsize=(12, 4))
    # DRIF
    activelattice = '../../element/lattices/misc/Drift-Bend-Quad/Drift.lte'
    latticedata = returnlatticedata(activelattice, 'felix_normal')
    trackingdata = returntrackingdata(latticedata, tracksett)
    ax = fig.add_subplot(131)
    plotphasespace('Drift space')

    # BEND
    activelattice = '../../element/lattices/misc/Drift-Bend-Quad/Bend.lte'
    latticedata = returnlatticedata(activelattice, 'felix_normal')
    trackingdata = returntrackingdata(latticedata, tracksett)
    ax = fig.add_subplot(132)
    plotphasespace('Dipole magnet')

    # Quad
    activelattice = '../../element/lattices/misc/Drift-Bend-Quad/Quad.lte'
    latticedata = returnlatticedata(activelattice, 'felix_normal')
    trackingdata = returntrackingdata(latticedata, tracksett)
    ax = fig.add_subplot(133)
    plotphasespace('Quadrupole magnet')

    plt.tight_layout()
    plt.savefig('../images/03-transformation-phase-space-ellipse.pdf')
    # plt.show()
