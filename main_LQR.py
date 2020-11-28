#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 16:41:02 2020

@author: Long Wang
"""
from datetime import date
import numpy as np
import matplotlib.pyplot as plt

from loss.lqr import LQR

from algorithm.spsa import SPSA
from algorithm.cs_spsa import CsSPSA

from utility.utility import Utility

np.random.seed(99)

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
a = 0.0001; A = 100; alpha = 0.602;
c = 0.1; gamma = 0.151
iter_num = 1000; rep_num = 1

### SPSA ###
print("running SPSA")
SPSA_optimizer = SPSA(a=a, A=A, alpha=alpha,
                      c=c, gamma=gamma,
                      iter_num=int(iter_num/2), rep_num=rep_num,
                      theta_0=theta_0, loss_obj=loss_obj,
                      record_theta_flag=True, record_loss_flag=True)

SPSA_optimizer.train()
SPSA_loss_per_iter = SPSA_optimizer.get_loss_per_iter()
SPSA_loss_per_iter_norm_error = Utility.get_loss_norm_error(SPSA_loss_per_iter, loss_0, loss_star)
Utility.plot_norm_error(SPSA_loss_per_iter_norm_error, "SPSA loss")

print("SPSA terminal loss error:", SPSA_loss_per_iter_norm_error[-1])
with open('data/LQR_SPSA_2020_11_27.npy', 'wb') as f:
    np.save(f, SPSA_loss_per_iter_norm_error)

### CS-SPSA ###
print("running CS-SPSA")
CS_SPSA_optimizer = CsSPSA(a=a, A=A, alpha=alpha,
                           c=c, gamma=gamma,
                           iter_num=iter_num, rep_num=rep_num,
                           theta_0=theta_0, loss_obj=loss_obj,
                           record_theta_flag=True, record_loss_flag=True)

CS_SPSA_optimizer.train()
CS_SPSA_loss_per_iter = CS_SPSA_optimizer.get_loss_per_iter()
CS_SPSA_loss_per_iter_norm_error = Utility.get_loss_norm_error(CS_SPSA_loss_per_iter, loss_0, loss_star)
Utility.plot_norm_error(CS_SPSA_loss_per_iter_norm_error, "CS-SPSA loss")

print("CS-SPSA terminal loss error:", CS_SPSA_loss_per_iter_norm_error[-1])
with open('data/LQR_CS_SPSA_2020_11_27.npy', 'wb') as f:
    np.save(f, CS_SPSA_loss_per_iter_norm_error)

### read SPSA and CS-SPSA ###
with open('data/LQR_SPSA_2020_11_27.npy', 'rb') as f:
    SPSA_loss_per_iter_norm_error = np.load(f)
with open('data/LQR_CS_SPSA_2020_11_27.npy', 'rb') as f:
    CS_SPSA_loss_per_iter_norm_error = np.load(f)

### read NPG and SUSD ###
# NPG
with open('data/LQR_PG_2020_07_19.npy', 'rb') as f:
    NPG_loss_per_iter = np.load(f)
NPG_loss_per_iter = NPG_loss_per_iter[0:5].reshape((5,1))
NPG_loss_per_iter_norm_error = Utility.get_loss_norm_error(NPG_loss_per_iter, loss_0, loss_star)
with open('data/LQR_SUSD_2020_07_19.npy', 'rb') as f:
    SUSD_loss_per_iter = np.load(f)
SUSD_loss_per_iter = SUSD_loss_per_iter[0:100].reshape((100,1))
SUSD_loss_per_iter_norm_error = Utility.get_loss_norm_error(SUSD_loss_per_iter, loss_0, loss_star)

### plot ###
today = date.today()

# plot loss
fig = plt.figure()
ax = fig.add_subplot()
plt.grid()
line_NPG, = ax.plot(np.concatenate(([1], np.repeat(NPG_loss_per_iter_norm_error, 200))), 'k:')
line_SUSD, = ax.plot(np.concatenate(([1], np.repeat(SUSD_loss_per_iter_norm_error, 10))), 'k-.')
line_SPSA, = ax.plot(np.concatenate(([1], np.repeat(SPSA_loss_per_iter_norm_error, 2))), 'k--', dashes=(5,5))
line_SPSA_for_legend, = ax.plot([1], 'k--')
line_CS_SPSA, = ax.plot(np.concatenate(([1], CS_SPSA_loss_per_iter_norm_error)), 'k-')

plt.xlim(xmin=0, xmax=iter_num)
plt.yscale("log")
plt.legend((line_NPG, line_SUSD, line_SPSA_for_legend, line_CS_SPSA),
           ("NPG", "SUSD", "SPSA", "CS-SPSA"), loc="best")
plt.xlabel("Number of Function Measurements")
plt.ylabel("Normalized Errors in Loss (log scale)")
plt.savefig("figure/LQR_p_" + str(p) + "_loss_" + str(today) + ".pdf", bbox_inches='tight')
plt.show()

### APL plot ###
plt.figure()
plt.grid()
plt.plot(np.concatenate(([1], np.repeat(NPG_loss_per_iter_norm_error, 200))))
plt.plot(np.concatenate(([1], np.repeat(SUSD_loss_per_iter_norm_error, 10))))
plt.plot(np.concatenate(([1], np.repeat(SPSA_loss_per_iter_norm_error, 2))))
plt.plot(np.concatenate(([1], CS_SPSA_loss_per_iter_norm_error)))

plt.xlim(xmin=0, xmax=iter_num)
plt.yscale("log")
plt.legend(("NPG", "SUSD", "SPSA", "CS-SPSA"), loc="best")
plt.xlabel("Number of Function Measurements")
plt.ylabel("Normalized Errors in Loss (log scale)")
plt.savefig("figure/LQR_p_" + str(p) + "_loss_" + str(today) + "_APL.pdf", bbox_inches='tight')
plt.show()
