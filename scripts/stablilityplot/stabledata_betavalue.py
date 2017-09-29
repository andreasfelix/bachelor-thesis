# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

print('Matplotlib version: {}'.format(mpl.__version__))

cmap = mpl.cm.Set1
SIZE_1 = 14
SIZE_2 = 16
SIZE_3 = 12

plt.rc('font', size=SIZE_1)  # controls default text sizes
plt.rc('axes', titlesize=SIZE_1)  # fontsize of the axes title
plt.rc('xtick', labelsize=SIZE_1)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE_1)  # fontsize of the tick labels
plt.rc('axes', labelsize=SIZE_2)  # fontsize of the x and y labels
plt.rc('legend', fontsize=SIZE_3)  # legend fontsize

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

    iterations = 500

    grid = np.zeros((iterations, iterations))

    k_values_1 = np.linspace(0, 3, iterations)
    k_values_2 = np.linspace(0, 3, iterations)

    t1 = time.clock()
    for i, k_value_1 in enumerate(k_values_1):
        for i2, k_value_2 in enumerate(k_values_2):
            LatticeEditor.set_value(activelattice, 'Q3T2', 'K1', -k_value_1)
            LatticeEditor.set_value(activelattice, 'Q4T2', 'K1', k_value_2)
            latticedata = returnlatticedata(activelattice, 'felix_twice')
            twissdata = returntwissdata(latticedata, beta=True, betatronphase=False, momentumcompaction=False)

            if twissdata.stable_x and twissdata.stable_y:
                # plt.plot(QF_K1, QD_K1, '.r')
                grid[i, i2] = np.max([twissdata.betax, twissdata.betay])

    print("time for loop", time.clock() - t1)

    pickle_dict = {"grid": grid, "k_values_1": k_values_1, "k_values_2": k_values_2}
    with open("../../../data/bessy2_Q3_vs_Q4_Q5_off_betavalue.pkl", "wb") as outputfile:
        pickle.dump(pickle_dict, outputfile)
