#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 16:41:02 2020

@author: Long Wang
"""
from datetime import date
import time

import numpy as np
import matplotlib.pyplot as plt

from loss.lqr import LQR

from algorithm.spsa import SPSA
from algorithm.cs_spsa import CsSPSA

from utility.utility import Utility

today = str(date.today())
np.random.seed(100)

# initialize loss object
p = 12
loss_obj = LQR(p=p)

K_0 = np.ones(loss_obj.K_star.shape) * 2
theta_0 = K_0.flatten()
theta_star = loss_obj.K_star.flatten()

loss_0 = loss_obj.get_loss_true(theta_0)
loss_star = 4149.38952236
print()
print("loss_0, loss_star:", loss_0, loss_star)
print()

# initialize optimizer parameters
a = 0.0001; A = 50; alpha = 0.668;
c = 0.5; gamma = 0.167
iter_num = 500; rep_num = 1

### SPSA ###
print("running SPSA")
SPSA_optimizer = SPSA(a=a, A=A, alpha=alpha,
                      c=c, gamma=gamma,
                      iter_num=iter_num, rep_num=rep_num,
                      theta_0=theta_0, loss_obj=loss_obj,
                      record_theta_flag=True, record_loss_flag=True)
start = time.time()
SPSA_optimizer.train()
end = time.time()
print("SPSA time:", end - start)

SPSA_loss = SPSA_optimizer.get_loss_per_iter()
SPSA_loss_norm_error = Utility.get_loss_norm_error(SPSA_loss, loss_0, loss_star)
Utility.plot(SPSA_loss_norm_error, "SPSA loss")

np.savez("data/LQR_SPSA_" + today + ".npz",
         SPSA_loss_norm_error = SPSA_loss_norm_error)
print("SPSA terminal loss error:", SPSA_loss_norm_error[-1])

### CS-SPSA ###
print("running CS-SPSA")
CS_SPSA_optimizer = CsSPSA(a=a, A=A, alpha=alpha,
                           c=c, gamma=gamma,
                           iter_num=iter_num, rep_num=rep_num,
                           theta_0=theta_0, loss_obj=loss_obj,
                           record_theta_flag=True, record_loss_flag=True)
start = time.time()
CS_SPSA_optimizer.train()
end = time.time()
print("CS_SPSA time:", end - start)

CS_SPSA_loss = CS_SPSA_optimizer.get_loss_per_iter()
CS_SPSA_loss_norm_error = Utility.get_loss_norm_error(CS_SPSA_loss, loss_0, loss_star)
Utility.plot(CS_SPSA_loss_norm_error, "CS-SPSA loss")

np.savez("data/LQR_CS_SPSA_" + today + ".npz",
         CS_SPSA_loss_norm_error = CS_SPSA_loss_norm_error)
print("CS_SPSA terminal loss error:", CS_SPSA_loss_norm_error[-1])

### read SPSA and CS-SPSA ###
# SPSA_norm_error = np.load("data/LQR_SPSA_2021-01-13.npz", allow_pickle = True)
# SPSA_loss_norm_error = SPSA_norm_error['SPSA_loss_norm_error']
# CS_SPSA_norm_error = np.load("data/LQR_CS_SPSA_2021-01-13.npz", allow_pickle = True)
# CS_SPSA_loss_norm_error = CS_SPSA_norm_error['CS_SPSA_loss_norm_error']

### read NPG and SUSD ###
# NPG
# with open('data/LQR_PG_2020_07_19.npy', 'rb') as f:
#     NPG_loss = np.load(f)
# NPG_loss = NPG_loss[0:5].reshape((5,1))
# NPG_loss_norm_error = Utility.get_loss_norm_error(NPG_loss, loss_0, loss_star)
# with open('data/LQR_SUSD_2020_07_19.npy', 'rb') as f:
#     SUSD_loss = np.load(f)
# SUSD_loss = SUSD_loss[0:100].reshape((100,1))
# SUSD_loss_norm_error = Utility.get_loss_norm_error(SUSD_loss, loss_0, loss_star)

### plot ###
today = date.today()

# plot loss
fig = plt.figure()
ax = fig.add_subplot()
plt.grid()
# line_NPG, = ax.plot(np.concatenate(([1], np.repeat(NPG_loss_norm_error, 100))), 'k:')
# line_SUSD, = ax.plot(np.concatenate(([1], np.repeat(SUSD_loss_norm_error, 5))), 'k-.')
line_SPSA, = ax.plot(np.concatenate(([1], SPSA_loss_norm_error)), 'k--', dashes=(5,5))
line_SPSA_for_legend, = ax.plot([1], 'k--')
line_CS_SPSA, = ax.plot(np.concatenate(([1], CS_SPSA_loss_norm_error)), 'k-')

plt.xlim(xmin=0, xmax=iter_num)
plt.yscale("log")
plt.legend((line_SPSA_for_legend, line_CS_SPSA),
           ("SPSA", "CS-SPSA"), loc="best")
plt.xlabel("Number of Iterations")
plt.ylabel("Normalized Errors in Loss (log scale)")
plt.savefig("figure/LQR_loss_" + str(today) + ".pdf")#, bbox_inches='tight')
plt.show()