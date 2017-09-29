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

from matplotlib import rc

# for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

import numpy
import sys
from os.path import expanduser
import time

sys.path.append(expanduser("~") + "/git")
import shutil
import pickle

from element.pyfiles.compute.transfermatrix_element_slicing import returnlatticedata
from element.pyfiles.compute.twissdata import returntwissdata
from element.pyfiles.miscellaneous.LatticeTools import LatticeEditor

if __name__ == '__main__':
    latticepath = 'BII_2017-03-28_17-54_LOCOFitByPS_noID_ActualUserMode_third_best_Q5T2off.lte'
    activelattice = 'active.lte'
    shutil.copy(latticepath, activelattice)

    iterations = 1600 * 2

    grid = np.zeros((iterations, iterations))

    k_values_1 = np.linspace(-4.0, 4.0, iterations)
    k_values_2 = np.linspace(-4.0, 4.0, iterations)

    t1 = time.clock()
    for i, k_value_1 in enumerate(k_values_1):
        for i2, k_value_2 in enumerate(k_values_2):
            LatticeEditor.set_value(activelattice, 'Q4T2', 'K1', k_value_1)
            LatticeEditor.set_value(activelattice, 'Q3T2', 'K1', -k_value_2)
            latticedata = returnlatticedata(activelattice, 'felix_once')
            twissdata = returntwissdata(latticedata, beta=False, betatronphase=False, momentumcompaction=False)

            if twissdata.stable_x and twissdata.stable_y:
                # plt.plot(QF_K1, QD_K1, '.r')
                grid[i, i2] = 1

    print("time for loop", time.clock() - t1)

    pickle_dict = {"grid": grid, "k_values_1": k_values_1, "k_values_2": k_values_2}
    with open("../../../data/bessy2_Q3_vs_Q4_Q5off.pkl", "wb") as outputfile:
        pickle.dump(pickle_dict, outputfile)
