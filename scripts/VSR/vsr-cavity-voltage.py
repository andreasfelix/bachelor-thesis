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

t = np.linspace(-1, 5, 1000)
fnc = 0.5
Vnc =  1.5 * np.sin(t * 2 * np.pi * fnc)
f1 = 1.5
V1 = 20 * np.sin(t * 2 * np.pi * f1)
f2 = 1.75
V2 = 17.14 * np.sin(t * 2 * np.pi * f2)
Vsum = Vnc + V1 + V2

fig = plt.figure(figsize=(9, 5))

plt.plot(t, Vnc, c='green', label='0.5 GHz')
plt.plot(t, V1, c='blue', label='1.5 GHz')
plt.plot(t, V2, c='red', label='1.75 GHz')
plt.plot(t, Vsum, c='black', label='Sum')
ms, mew = 14, 4
plt.plot([0, 4], [0, 0], 'o', markerfacecolor="None", ms=ms, markeredgecolor='blue', markeredgewidth=mew, label="short bunches")
plt.plot(2, 0, 'o', markerfacecolor="None", ms=ms, markeredgecolor='red', markeredgewidth=mew, label="long bunches")

plt.xlabel('time $t$ / ns')
plt.ylabel('voltage $V$ / MV')
plt.legend(loc='upper right')
plt.xlim(t[0], t[-1])
plt.ylim(-40, 40)

plt.tight_layout()
plt.savefig("../../images/01-cavity-voltage.pdf", transparent=True)
