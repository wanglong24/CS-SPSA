#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Long Wang
"""
from datetime import date
import numpy as np
import matplotlib.pyplot as plt

from loss.Gaussian_signal_noise import GaussianSignalNoise

from algorithm.spsa import SPSA
from algorithm.cs_spsa import CsSPSA

from utility.utility import Utility

np.random.seed(99)

# initialize loss object
p = 10; n = 100
loss_obj = GaussianSignalNoise(p=p, n=n)

Sigma_0 = np.eye(p)
theta_0 = Sigma_0[np.tril_indices(p)]
theta_0 += np.random.rand(theta_0.size)
Sigma_0 = loss_obj.convert_theta_to_Sigma(theta_0)

Sigma_0_norm_error = np.linalg.norm(Sigma_0 - loss_obj.Sigma, ord="fro")
loss_0 = loss_obj.get_loss_true(theta_0)
loss_star = loss_obj.get_loss_true_mu_Sigma(loss_obj.mu, loss_obj.Sigma)
print()
print("loss_0, loss_star:", loss_0, loss_star)
print()

# initialize optimizer parameters
a = 0.5; A = 100; alpha = 0.602
c = 0.05; gamma = 0.151
iter_num = 1000; rep_num = 10

### SGD ###
print("running SGD")
SGD_loss_per_iter = np.empty(iter_num)
SGD_Sigma_per_iter_norm_error = np.empty(iter_num)
Sigma_k = Sigma_0
for iter_idx in range(iter_num):
    a_k = 0.01 / (iter_idx + 1 + A) ** alpha
    grad_Sigma_k = loss_obj.get_grad_Sigma_noisy(iter_idx, loss_obj.mu, Sigma_k)
    Sigma_k -= a_k * grad_Sigma_k

    SGD_loss_per_iter[iter_idx] = loss_obj.get_loss_true_mu_Sigma(loss_obj.mu, Sigma_k)
    SGD_Sigma_per_iter_norm_error[iter_idx] = np.linalg.norm(Sigma_k - loss_obj.Sigma, ord="fro") / Sigma_0_norm_error
SGD_loss_per_iter_norm_error = (SGD_loss_per_iter - loss_star) / (loss_0 - loss_star)

Utility.plot_norm_error(SGD_loss_per_iter_norm_error, "SGD loss")

print("SGD terminal loss error:", SGD_loss_per_iter_norm_error[-1])

### SPSA ###
# print("running SPSA")
# SPSA_optimizer = SPSA(a=a, A=A, alpha=alpha,
#                       c=c, gamma=gamma,
#                       iter_num=iter_num, rep_num=rep_num,
#                       theta_0=theta_0, loss_obj=loss_obj,
#                       record_theta_flag=True, record_loss_flag=True)

# SPSA_optimizer.train()
# SPSA_loss_per_iter = SPSA_optimizer.get_loss_per_iter()
# SPSA_loss_per_iter_norm_error = Utility.get_loss_norm_error(SPSA_loss_per_iter, loss_0, loss_star)
# Utility.plot_norm_error(SPSA_loss_per_iter_norm_error, "SPSA loss")

# print("SPSA terminal loss error:", SPSA_loss_per_iter_norm_error[-1])

### CS-SPSA ###
print("running CS-SPSA")
CS_SPSA_optimizer = CsSPSA(a=0.05, A=100, alpha=alpha,
                           c=0.05, gamma=gamma,
                           iter_num=iter_num, rep_num=rep_num,
                           theta_0=theta_0, loss_obj=loss_obj,
                           record_theta_flag=True, record_loss_flag=True)

CS_SPSA_optimizer.train()
CS_SPSA_loss_per_iter = CS_SPSA_optimizer.get_loss_per_iter()
CS_SPSA_loss_per_iter_norm_error = Utility.get_loss_norm_error(CS_SPSA_loss_per_iter, loss_0, loss_star)
Utility.plot_norm_error(CS_SPSA_loss_per_iter_norm_error, "CS-SPSA loss")

print("CS-SPSA terminal loss error:", CS_SPSA_loss_per_iter_norm_error[-1])

### plot ###
today = date.today()

# plot loss
plt.figure()
plt.grid()
plt.plot(np.concatenate(([1], SGD_loss_per_iter_norm_error)), 'k:')
plt.plot(np.concatenate(([1], CS_SPSA_loss_per_iter_norm_error)), 'k-')

plt.xlim(xmin=0, xmax=iter_num)
plt.yscale("log")
plt.legend(("SGD", "CS-SPSA"), loc="best")
plt.xlabel("Number of Iterations")
plt.ylabel("Normalized Errors in Loss (log scale)")
plt.savefig("figure/Gaussian_signal_noise_p_" + str(p) + "_loss_" + str(today) + ".pdf", bbox_inches='tight')
plt.show()

### APL plot ###
plt.figure()
plt.grid()
plt.plot(np.concatenate(([1], SGD_loss_per_iter_norm_error)))
plt.plot(np.concatenate(([1], CS_SPSA_loss_per_iter_norm_error)))

plt.xlim(xmin=0, xmax=iter_num)
plt.yscale("log")
plt.legend(("SGD", "CS-SPSA"), loc="best")
plt.xlabel("Number of Iterations")
plt.ylabel("Normalized Errors in Loss (log scale)")
plt.savefig("figure/Gaussian_signal_noise_p_" + str(p) + "_loss_" + str(today) + "_APL.pdf", bbox_inches='tight')
plt.show()

print("finish plotting")