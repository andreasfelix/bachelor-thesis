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
import pandas as pd

if __name__ == '__main__':
    # setup
    t1 = time.clock()
    iterations = 30
    k_start_list = [0, 0, 0, 0, 0.15, 0.15, 0.15, 0.15]
    k_end_list = [0.15, 0.25, 0.5, 0.75, 0.2, 0.25, 0.3, 0.5]
    Nges2_list = []
    Nges4_list = []
    N2_list = []
    N4_list = []
    step_list = []

    for k_start, k_end in zip(k_start_list, k_end_list):
        k_values, step = np.linspace(k_start, k_end, iterations, retstep=True)
        step_list.append(step)
        print("K_values:", k_start, k_end)

        Nges2 = iterations ** 2
        Nges4 = iterations ** 4
        Nges2_list.append(Nges2)
        Nges4_list.append(Nges4)

        # fodo two PS
        N2 = 0
        latticepath = 'fodoscan_two_PS.lte'
        activelattice = 'active.lte'
        shutil.copy(latticepath, activelattice)

        for i, k_value_1 in enumerate(k_values):
            for i2, k_value_2 in enumerate(k_values):
                LatticeEditor.set_value(activelattice, 'QF', 'K1', k_value_1)
                LatticeEditor.set_value(activelattice, 'QD', 'K1', -k_value_2)
                latticedata = returnlatticedata(activelattice, 'felix_once')
                twissdata = returntwissdata(latticedata, beta=False, betatronphase=False, momentumcompaction=False)

                if twissdata.stable_x and twissdata.stable_y:
                    N2 += 1

        # fodo two PS
        N4 = 0
        latticepath = 'fodoscan_four_PS.lte'
        activelattice = 'active.lte'
        shutil.copy(latticepath, activelattice)

        for i, k_value_1 in enumerate(k_values):
            # print("Iteration {} of {}".format(i, iterations))
            for i2, k_value_2 in enumerate(k_values):
                for i3, k_value_3 in enumerate(k_values):
                    for i4, k_value_4 in enumerate(k_values):
                        LatticeEditor.set_value(activelattice, 'QF', 'K1', k_value_1)
                        LatticeEditor.set_value(activelattice, 'QD', 'K1', -k_value_2)
                        LatticeEditor.set_value(activelattice, 'QF2', 'K1', k_value_3)
                        LatticeEditor.set_value(activelattice, 'QD2', 'K1', -k_value_4)
                        latticedata = returnlatticedata(activelattice, 'felix_once')
                        twissdata = returntwissdata(latticedata, beta=False, betatronphase=False, momentumcompaction=False)

                        if twissdata.stable_x and twissdata.stable_y:
                            N4 += 1

        N2_list.append(N2)
        N4_list.append(N4)

        print("Total Time:", time.clock() - t1)

        # print("N", N2, N4)
        # print("Nges", Nges2, Nges4)
        # print("N / Nges", N2 / Nges2, N4 / Nges4)

    df = pd.DataFrame([k_start_list, k_end_list, step_list, N2_list, N4_list, Nges2_list, Nges4_list])
    df = df.transpose()
    df.columns = ['$k_{\\textup{start}}$', '$k_{\\textup{end}}$', '$\\Delta k$', '$N_2$', '$N_4$', '$N_{\\textup{ges,2}}$', '$N_{\\textup{ges,4}}$']

    df['$n_2$'] = df['$N_2$'] / df['$N_{\\textup{ges,2}}$']
    df['$n_4$'] = df['$N_4$'] / df['$N_{\\textup{ges,4}}$']
    df['$\\frac{n_2}{n_4}$'] = df['$n_2$'] / df['$n_4$']
    df.to_csv("scanproportian.csv", index=False)

    df['$\\Delta k$'] = df['$\\Delta k$'].round()
    df['$n_2$'] = df['$n_2$'].round(2)
    df['$n_4$'] = df['$n_4$'].round(2)
    df['$\\frac{n_2}{n_4}$'] = df['$\\frac{n_2}{n_4}$'].round(2)
    df.to_latex("scanproportian.tex", index=False, escape=False)
    df[['$k_{\\textup{start}}$', '$k_{\\textup{end}}$', '$\\Delta k$', '$n_2$', '$n_4$', '$\\frac{n_2}{n_4}$']].to_latex("scanproportian_notall.tex", index=False, escape=False)
    print("TOTAL TIME FOR ALL: {} s".format(time.clock() - t1))
    print("TOTAL TIME FOR ALL: {} h".format((time.clock() - t1) / 3600))
