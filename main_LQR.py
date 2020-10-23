#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 16:41:02 2020

@author: Long Wang
"""

import numpy as np
import matplotlib.pyplot as plt

from loss.lqr import LQR

from algorithm.spsa import SPSA
from algorithm.cs_spsa import CsSPSA

np.random.seed(100)

p = 12; T = 100
n = 4; m = 3
x_0 = 20 * np.array([1, 2, -1, -0.5]).reshape(n,1)

LQR_model = LQR(p=p, T=T, x_0=x_0)

def loss_true(K):
    return LQR_model.compute_cost(K)

def loss_noisy(K):
    return LQR_model.compute_cost_noisy(K)

def get_normalize_loss_error(loss_all, loss_0, loss_star, multiplier):
    loss_ks = np.mean(loss_all, axis=1) # average over replicates
    loss_ks_error = (loss_ks-loss_star) / (loss_0-loss_star)
    return np.repeat(loss_ks_error, multiplier)

K_star = np.array([
    [1.60233232e-01, -1.36227805e-01, -9.93576677e-02, -4.28244630e-02],
    [7.47596033e-02,  9.05753832e-02,  7.46951286e-02, -1.53947620e-01],
    [3.65372978e-01, -2.59862175e-04,  5.91522023e-02, 8.25660846e-01]])

theta_star = K_star.flatten()
# loss_star = loss_true(theta_star)
loss_star = 4149.38952236

K_0 = np.ones(K_star.shape) * 2
theta_0 = K_0.flatten()
loss_0 = loss_true(theta_0)
print("loss_0", loss_0)

# parameters
a = 0.0001; c = 0.1; A = 100
alpha = 0.602; gamma = 0.151
iter_num = 1000; rep_num = 20

print("running SPSA")
SPSA_solver = SPSA(a=a, c=c, A=A, alpha=alpha, gamma=gamma,
                    iter_num=int(iter_num/2), rep_num=rep_num,
                    theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy)
# SPSA_solver.train()
# with open('LQR_SPSA_2020_09_18.npy', 'wb') as f:
    # np.save(f, SPSA_solver.loss_k_all)

# SPSA_loss_ks_error = get_normalize_loss_error(SPSA_solver.loss_k_all, loss_0, loss_star, 2)
# plt.figure()
# plt.grid()
# plt.yscale("log")
# plt.plot(np.concatenate(([1], SPSA_loss_ks_error)), 'k-.')

print("running CS-SPSA")
CS_SPSA_solver = CsSPSA(a=a, c=c, A=A, alpha=alpha, gamma=gamma,
                   iter_num=iter_num, rep_num=rep_num,
                   theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy)
# CS_SPSA_solver.train()
# with open('LQR_CS_SPSA_2020_09_18.npy', 'wb') as f:
    # np.save(f, CS_SPSA_solver.loss_k_all)

# CS_SPSA_loss_ks_error = get_normalize_loss_error(CS_SPSA_solver.loss_k_all, loss_0, loss_star, 1)
# plt.figure()
# plt.grid()
# plt.plot(np.concatenate(([1], CS_SPSA_loss_ks_error)), 'k-')

# SPSA_loss_ks_error = get_normalize_loss_error(SPSA_solver.loss_k_all, loss_0, loss_star, 2)
# CS_SPSA_loss_ks_error = get_normalize_loss_error(CS_SPSA_solver.loss_k_all, loss_0, loss_star, 1)

with open('LQR_SPSA_2020_09_18.npy', 'rb') as f:
    SPSA_loss_k_all = np.load(f, allow_pickle=True)
with open('LQR_CS_SPSA_2020_09_18.npy', 'rb') as f:
    CS_SPSA_loss_k_all = np.load(f, allow_pickle=True)

SPSA_loss_ks_error = get_normalize_loss_error(SPSA_loss_k_all, loss_0, loss_star, 2)
CS_SPSA_loss_ks_error = get_normalize_loss_error(CS_SPSA_loss_k_all, loss_0, loss_star, 1)


# NPG
with open('LQR_PG_2020_07_19.npy', 'rb') as f:
    NPG_loss_k_all = np.load(f)
NPG_loss_k_all = NPG_loss_k_all[0:5].reshape((5,1))
NPG_loss_ks_error = get_normalize_loss_error(NPG_loss_k_all, loss_0, loss_star, 200)

# SUSD
with open('LQR_SUSD_2020_07_19.npy', 'rb') as f:
    SUSD_loss_k_all = np.load(f)
SUSD_loss_k_all = SUSD_loss_k_all[0:100].reshape((100,1))
SUSD_loss_ks_error = get_normalize_loss_error(SUSD_loss_k_all, loss_0, loss_star, 10)

plt.figure()
plt.grid()

plt.plot(np.concatenate(([1], NPG_loss_ks_error)), 'k:')
plt.plot(np.concatenate(([1], SUSD_loss_ks_error)), 'k-.')
plt.plot(np.concatenate(([1], SPSA_loss_ks_error)), 'k--')
plt.plot(np.concatenate(([1], CS_SPSA_loss_ks_error)), 'k-')

plt.xlim(xmin=0, xmax=iter_num)
plt.yscale("log")
plt.legend(["NPG", "SUSD", "SPSA", "CS-SPSA"])
plt.xlabel("Number of Function Measurements")
plt.ylabel("Normalized Error in Loss Function")
plt.savefig('figure/LQR_loss_2020_09_18.pdf')
plt.show()