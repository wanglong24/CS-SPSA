from datetime import date
import time

import numpy as np
import matplotlib.pyplot as plt

from loss.quadratic_exponential import QuadraticExponential

from algorithm.fdsa import FDSA
from algorithm.spsa import SPSA
from algorithm.cs_fdsa import CsFDSA
from algorithm.cs_spsa import CsSPSA

from utility.utility import Utility

today = str(date.today())
np.random.seed(99)

# initialize loss object
p = 100
loss_obj = QuadraticExponential(p)

theta_0 = np.ones(p)
loss_0 = loss_obj.get_loss_true(theta_0)
theta_star = loss_obj.get_theta_star()
loss_star = loss_obj.get_loss_true(theta_star)
theta_0_norm_error = np.linalg.norm(theta_0 - theta_star)

# parameters
if p == 10:
    a = 0.1; c = 0.5; A = 10
    iter_num = 5000
elif p == 100:
    a = 0.1; c = 0.5; A = 100
    iter_num = 5000
else:
    a = 0; c = 0; A = 0

alpha = 0.7; gamma = 0.18
rep_num = 20

### FDSA ###
print("running FDSA")
FDSA_optimizer = FDSA(a=a, A=A, alpha=alpha,
                      c=c, gamma=gamma,
                      iter_num=int(iter_num/p), rep_num=rep_num,
                      theta_0=theta_0, loss_obj=loss_obj,
                      record_theta_flag=True, record_loss_flag=True)
start = time.time()
FDSA_optimizer.train()
end = time.time()
print(end - start)

FDSA_theta = FDSA_optimizer.get_theta_per_iter()
FDSA_theta_norm_error = Utility.get_theta_norm_error(FDSA_theta, theta_0, theta_star)
# Utility.plot(FDSA_theta_norm_error, "FDSA theta")

FDSA_loss = FDSA_optimizer.get_loss_per_iter()
FDSA_loss_norm_error = Utility.get_loss_norm_error(FDSA_loss, loss_0, loss_star)
# Utility.plot(FDSA_loss_norm_error, "FDSA loss")

np.savez("data/Quad_Exp_FDSA_p_" + str(p) + "_"+ today + ".npz",
         FDSA_theta_norm_error = FDSA_theta_norm_error,
         FDSA_loss_norm_error = FDSA_loss_norm_error)
print("FDSA terminal loss error:", FDSA_loss_norm_error[-1])

### CS-FDSA ###
print("running CS_FDSA")
CS_FDSA_optimizer = CsFDSA(a=a, A=A, alpha=alpha,
                           c=c, gamma=gamma,
                           iter_num=int(iter_num/p), rep_num=rep_num,
                           theta_0=theta_0, loss_obj=loss_obj,
                           record_theta_flag=True, record_loss_flag=True)
start = time.time()
CS_FDSA_optimizer.train()
end = time.time()
print(end - start)

CS_FDSA_theta = CS_FDSA_optimizer.get_theta_per_iter()
CS_FDSA_theta_norm_error = Utility.get_theta_norm_error(CS_FDSA_theta, theta_0, theta_star)
# Utility.plot(CS_FDSA_theta_norm_error, "CS_FDSA theta")

CS_FDSA_loss = CS_FDSA_optimizer.get_loss_per_iter()
CS_FDSA_loss_norm_error = Utility.get_loss_norm_error(CS_FDSA_loss, loss_0, loss_star)
# Utility.plot(CS_FDSA_loss_norm_error, "CS_FDSA loss")

np.savez("data/Quad_Exp_CS_FDSA_p_" + str(p) + "_"+ today + ".npz",
         CS_FDSA_theta_norm_error = CS_FDSA_theta_norm_error,
         CS_FDSA_loss_norm_error = CS_FDSA_loss_norm_error)
print("CS_FDSA terminal loss error:", CS_FDSA_loss_norm_error[-1])

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
print(end - start)

SPSA_theta = SPSA_optimizer.get_theta_per_iter()
SPSA_theta_norm_error = Utility.get_theta_norm_error(SPSA_theta, theta_0, theta_star)
# Utility.plot(SPSA_theta_norm_error, "SPSA theta")

SPSA_loss = SPSA_optimizer.get_loss_per_iter()
SPSA_loss_norm_error = Utility.get_loss_norm_error(SPSA_loss, loss_0, loss_star)
# Utility.plot(SPSA_loss_norm_error, "SPSA loss")

np.savez("data/Quad_Exp_SPSA_p_" + str(p) + "_"+ today + ".npz",
         SPSA_theta_norm_error = SPSA_theta_norm_error,
         SPSA_loss_norm_error = SPSA_loss_norm_error)
print("SPSA terminal loss error:", SPSA_loss_norm_error[-1])

### CS-SPSA ###
print("running CS_SPSA")
CS_SPSA_optimizer = CsSPSA(a=a, A=A, alpha=alpha,
                           c=c, gamma=gamma,
                           iter_num=iter_num, rep_num=rep_num,
                           theta_0=theta_0, loss_obj=loss_obj,
                           record_theta_flag=True, record_loss_flag=True)
start = time.time()
CS_SPSA_optimizer.train()
end = time.time()
print(end - start)

CS_SPSA_theta = CS_SPSA_optimizer.get_theta_per_iter()
CS_SPSA_theta_norm_error = Utility.get_theta_norm_error(CS_SPSA_theta, theta_0, theta_star)
# Utility.plot(CS_SPSA_theta_norm_error, "CS_SPSA theta")

CS_SPSA_loss = CS_SPSA_optimizer.get_loss_per_iter()
CS_SPSA_loss_norm_error = Utility.get_loss_norm_error(CS_SPSA_loss, loss_0, loss_star)
# Utility.plot(CS_SPSA_loss_norm_error, "CS_SPSA loss")

np.savez("data/Quad_Exp_CS_SPSA_p_" + str(p) + "_"+ today + ".npz",
         CS_SPSA_theta_norm_error = CS_SPSA_theta_norm_error,
         CS_SPSA_loss_norm_error = CS_SPSA_loss_norm_error)
print("CS_SPSA terminal loss error:", CS_SPSA_loss_norm_error[-1])



### load data ###
# FDSA_norm_error = np.load("data/Quad_Exp_FDSA_p_" + str(p) + "_"+ today + ".npz", allow_pickle = True)
# FDSA_theta_norm_error = FDSA_norm_error['FDSA_theta_norm_error']
# FDSA_loss_norm_error = FDSA_norm_error['FDSA_loss_norm_error']

# CS_FDSA_norm_error = np.load("data/Quad_Exp_CS_FDSA_p_" + str(p) + "_"+ today + ".npz", allow_pickle = True)
# CS_FDSA_theta_norm_error = CS_FDSA_norm_error['CS_FDSA_theta_norm_error']
# CS_FDSA_loss_norm_error = CS_FDSA_norm_error['CS_FDSA_loss_norm_error']

# SPSA_norm_error = np.load("data/Quad_Exp_SPSA_p_" + str(p) + "_"+ today + ".npz", allow_pickle = True)
# SPSA_theta_norm_error = SPSA_norm_error['SPSA_theta_norm_error']
# SPSA_loss_norm_error = SPSA_norm_error['SPSA_loss_norm_error']

# CS_SPSA_norm_error = np.load("data/Quad_Exp_CS_SPSA_p_" + str(p) + "_"+ today + ".npz", allow_pickle = True)
# CS_SPSA_theta_norm_error = CS_SPSA_norm_error['CS_SPSA_theta_norm_error']
# CS_SPSA_loss_norm_error = CS_SPSA_norm_error['CS_SPSA_loss_norm_error']

### plot ###
# plot loss
fig = plt.figure()
ax = fig.add_subplot()
plt.grid()
line_FDSA, = ax.plot(np.concatenate(([1], np.repeat(FDSA_loss_norm_error, int(p)))), 'k:')
line_CS_FDSA, = ax.plot(np.concatenate(([1], np.repeat(CS_FDSA_loss_norm_error, int(p)))), 'k-.')
line_SPSA, = ax.plot(np.concatenate(([1], np.repeat(SPSA_loss_norm_error, int(1)))), 'k--', dashes=(5,5))
line_SPSA_for_legend, = ax.plot([1], 'k--')
line_CS_SPSA, = ax.plot(np.concatenate(([1], np.repeat(CS_SPSA_loss_norm_error, int(1)))), 'k-')

plt.xlim(xmin=0, xmax=iter_num)
plt.yscale("log")
plt.legend((line_FDSA, line_CS_FDSA, line_SPSA_for_legend, line_CS_SPSA),
           ("FDSA", "CS-FDSA", "SPSA", "CS-SPSA"), loc="best")
plt.xlabel("Normalized Number of Iterations")
plt.ylabel("Normalized Errors in Loss (log scale)")
plt.savefig("figure/Quad_Exp_loss_norm_error_p_" + str(p) + "_" + str(today) + ".pdf")
plt.show()

# plot theta
fig = plt.figure()
ax = fig.add_subplot()
plt.grid()
line_FDSA, = ax.plot(np.concatenate(([1], np.repeat(FDSA_theta_norm_error, int(2*p)))), 'k:')
line_CS_FDSA, = ax.plot(np.concatenate(([1], np.repeat(CS_FDSA_theta_norm_error, int(p)))), 'k-.')
line_SPSA, = ax.plot(np.concatenate(([1], np.repeat(SPSA_theta_norm_error, int(2)))), 'k--', dashes=(10,10))
line_SPSA_for_legend, = ax.plot([1], 'k--')
line_CS_SPSA, = ax.plot(np.concatenate(([1], np.repeat(CS_SPSA_theta_norm_error, int(1)))), 'k-')

plt.xlim(xmin=0, xmax=iter_num)
plt.yscale("log")
plt.legend((line_FDSA, line_CS_FDSA, line_SPSA_for_legend, line_CS_SPSA),
           ("FDSA", "CS-FDSA", "SPSA", "CS-SPSA"), loc="best")
plt.xlabel("Normalized Number of Iterations")
plt.ylabel(r"Normalized Errors in Estimate of $\mathbf{\theta}$ (log scale)")
plt.savefig("figure/Quad_Exp_theta_norm_error_p_" + str(p) + "_" + str(today) + ".pdf")
plt.show()

print("finish plotting")