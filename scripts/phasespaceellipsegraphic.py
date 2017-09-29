# -*- coding: utf-8 -*-
from  __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

print('Matplotlib version: {}'.format(mpl.__version__))

from matplotlib import rc

# for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

myred = '#c1151a'

cmap = mpl.cm.spectral

SMALL_SIZE = 22
MEDIUM_SIZE = 22
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize

import sys
from os.path import expanduser

sys.path.append(expanduser("~") + "/git")
from element.pyfiles.compute.transfermatrix_element_slicing import returnlatticedata
from element.pyfiles.compute.tracking import returntrackingdata
from element.pyfiles.compute.twisspara import returntwissdata
from element.pyfiles.visualize import latticeplot


class AttrClass:
    pass


def particle_distribution(N, emittance):
    X0 = np.zeros((5, N))
    phi = np.linspace(0, 2 * np.pi, N).reshape(1, N)
    X0[0, :] = emittance ** 0.5 * twissdata.betax[0] ** 0.5 * np.cos(phi)
    X0[1, :] = emittance ** 0.5 / twissdata.betax[0] ** 0.5 * np.sin(phi)
    X0[2:5, :] = np.zeros((3, N))
    return X0


def findnearest(array, value, idx):
    sign = np.sign(array[0] - value)
    for i, x in enumerate(array):
        if np.sign(x - value) != sign:
            sign = np.sign(array[i] - value)
            if trackingdata.dxdstrack[idx, i] >= 0:
                break
    if array[i] - value < value - array[i - 1]:
        return i
    else:
        return i - 1


if __name__ == '__main__':
    activelattice = '../../element/lattices/fodo-Cell/FODOCell_Wille_no_edge-focussing.lte'
    latticedata = returnlatticedata(activelattice, 'felix_full')

    twissdata = returntwissdata(latticedata)

    emittance = 5e-9
    tracksett = AttrClass
    tracksett.rounds = 1
    tracksett.N = 3000
    tracksett.X0_width = (1e-3, 1e-3, 0, 0, 0)
    tracksett.X0 = particle_distribution(tracksett.N, emittance)
    tracksett.random = False
    trackingdata = returntrackingdata(latticedata, tracksett)
    xlim_max = 1.5 * np.max(trackingdata.xtrack)
    ylim_max = 1.5 * np.max(trackingdata.dxdstrack)
    sc = ylim_max / xlim_max


    def plot(idx,num):
        plt.plot(trackingdata.xtrack[idx, :], trackingdata.dxdstrack[idx, :], '-', color='black')  # , linewidth=2.5)
        plt.plot(trackingdata.xtrack[idx, 0], trackingdata.dxdstrack[idx, 0], 'o', color=myred, zorder=10, markersize=8)

        plt.gca().set_xlim(-xlim_max, xlim_max)
        plt.gca().set_ylim(-ylim_max, ylim_max)

        if True:
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.xticks([])
            plt.yticks([])
            # plt.yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
            params = {'clip_on': False, 'fc': 'black'}
            width, head_width, head_length = 0.0000015, 0.00001, 0.00002
            ax.annotate('$u$', xy=(0.75 * xlim_max, 0.05 * ylim_max))
            ax.annotate('$u\'$', xy=(0.05 * xlim_max, 0.75 * ylim_max))

            plt.gca().add_patch(mpl.patches.FancyArrow(-0.75 * xlim_max, 0, 1.5 * xlim_max, 0, width=sc * width, head_width=sc * head_width, head_length=head_length, **params))
            plt.gca().add_patch(mpl.patches.FancyArrow(0, -0.75 * ylim_max, 0, 1.5 * ylim_max, width=width, head_width=head_width, head_length=sc * head_length, **params))

        # annotate psi
        xpos = xlim_max / 2
        ypos = -ylim_max / 4
        plt.plot([xlim_max / 10, xpos], [- ylim_max / 30, ypos], '-', color='black')
        plt.gca().annotate('$\\psi(s_{}) + \\psi_0$'.format(num), xy=(xpos, ypos), xytext=(xpos + xlim_max / 20, ypos - ylim_max / 20), fontsize=22)

        # annotate -alpha \sqrt \epsilon / beta
        anno_idx = np.argmax(trackingdata.xtrack[idx, :])
        xpos = trackingdata.xtrack[idx, anno_idx]
        ypos = trackingdata.dxdstrack[idx, anno_idx]
        plt.plot(xpos, ypos, 'o', color='black', zorder=10, markersize=6)
        plt.plot([xpos, xpos + xlim_max / 8], [ypos] * 2, '--', color='black')
        plt.gca().annotate('$-\\alpha \\sqrt{\\frac{\\epsilon}{\\beta}}$', xy=(xpos, ypos), xytext=(xpos + xlim_max / 6, ypos - ylim_max / 20), fontsize=22)
        plt.plot([xpos] * 2, [ypos, ypos - ylim_max / 5], '--', color='black')
        plt.gca().annotate('$\\sqrt{\\epsilon \\beta}$', xy=(xpos, ypos), xytext=(xpos - xlim_max / 10, ypos - ylim_max / 3.3), fontsize=22)

        # annotate -alpha \sqrt \epsilon / gamma
        anno_idx = np.argmax(trackingdata.dxdstrack[idx, :])
        xpos = trackingdata.xtrack[idx, anno_idx]
        ypos = trackingdata.dxdstrack[idx, anno_idx]
        plt.plot(xpos, ypos, 'o', color='black', zorder=10, markersize=6)
        plt.plot([xpos, xpos - xlim_max / 4.5], [ypos] * 2, '--', color='black')
        plt.gca().annotate('$\\sqrt{\\epsilon \\gamma}$', xy=(xpos, ypos), xytext=(xpos - xlim_max / 2.5, ypos - ylim_max / 20), fontsize=22)
        plt.plot([xpos] * 2, [ypos, ypos + ylim_max / 6], '--', color='black')
        plt.gca().annotate('$-\\alpha \\sqrt{\\frac{\\epsilon}{\\gamma}}$', xy=(xpos, ypos), xytext=(xpos - xlim_max / 10, ypos + ylim_max / 5), fontsize=22)

        # annotate epsilon beta
        anno_idx = findnearest(trackingdata.dxdstrack[idx, :], 0, idx)
        xpos = trackingdata.xtrack[idx, anno_idx]
        ypos = trackingdata.dxdstrack[idx, anno_idx]
        plt.plot(xpos, ypos, 'o', color='black', zorder=10, markersize=6)
        plt.gca().annotate('$\\sqrt{\\frac{\\epsilon}{\\gamma}}$', xy=(xpos, ypos), xytext=(xpos - xlim_max / 20, ypos + ylim_max / 12), fontsize=22)

        # annotate epsilon gamma
        anno_idx = findnearest(trackingdata.xtrack[idx, :], 0, idx)
        xpos = trackingdata.xtrack[idx, anno_idx]
        ypos = trackingdata.dxdstrack[idx, anno_idx]
        plt.plot(xpos, ypos, 'o', color='black', zorder=10, markersize=6)
        plt.gca().annotate('$\\sqrt{\\frac{\\epsilon}{\\beta}}$', xy=(xpos, ypos), xytext=(xpos - xlim_max / 5.2, ypos - ylim_max / 17), fontsize=22)


    fig = plt.figure('Tracking', figsize=(18, 8), facecolor='white', frameon=False)
    # left graphic
    ax = fig.add_subplot(121)
    pos_idx = np.searchsorted(latticedata.Cs, 1.5)
    plot(pos_idx,1)

    plt.gca().add_patch(mpl.patches.Arc([0, 0], 0.000075, sc * 0.000075, 0, -52, 0, color='black'))
    plt.plot([0, trackingdata.xtrack[pos_idx, 0]], [0, trackingdata.dxdstrack[pos_idx, 0]], '--', color='black')  # , linewidth=2.5)

    # right graphic
    ax = fig.add_subplot(122)
    pos_idx = np.searchsorted(latticedata.Cs, 5.5)
    plot(pos_idx,2)

    plt.gca().add_patch(mpl.patches.Arc([0, 0], 0.000075, sc * 0.000075, 0, -117, 0, color='black'))
    plt.plot([0, trackingdata.xtrack[pos_idx, 0]], [0, trackingdata.dxdstrack[pos_idx, 0]], '--', color='black')  # , linewidth=2.5)

    plt.tight_layout()
    # plt.show()
    plt.savefig('../images/03-phase-space-ellipse-graphic.pdf')
