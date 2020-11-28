from datetime import date
import numpy as np
import matplotlib.pyplot as plt

from loss.quadratic_exponential import QuadraticExponential

from algorithm.fdsa import FDSA
from algorithm.spsa import SPSA
from algorithm.cs_fdsa import CsFDSA
from algorithm.cs_spsa import CsSPSA

from utility.utility import Utility

np.random.seed(99)

# initialize loss object
p = 10
loss_obj = QuadraticExponential(p)

theta_0 = np.ones(p)
loss_0 = loss_obj.get_loss_true(theta_0)
theta_star = loss_obj.get_theta_star()
loss_star = loss_obj.get_loss_true(theta_star)
theta_0_norm_error = np.linalg.norm(theta_0 - theta_star)

# parameters
if p == 10:
    a = 0.02; c = 0.2; A = 100
elif p == 100:
    a = 0.01; c = 0.1; A = 100
else:
    a = 0; c = 0; A = 0

alpha = 0.602; gamma = 0.151
iter_num = 50000; rep_num = 20

### FDSA ###
print("running FDSA")
FDSA_optimizer = FDSA(a=a, A=A, alpha=alpha,
                      c=c, gamma=gamma,
                      iter_num=int(iter_num/(2*p)), rep_num=rep_num,
                      theta_0=theta_0, loss_obj=loss_obj,
                      record_theta_flag=True, record_loss_flag=True)
FDSA_optimizer.train()
# with open('data/Quadratic_exponential_p_10_FDSA_2020_11_26.npy', 'wb') as f:
#     np.save(f, FDSA_optimizer)

FDSA_theta_per_iter = FDSA_optimizer.get_theta_per_iter()
FDSA_theta_per_iter_norm_error = Utility.get_theta_norm_error(FDSA_theta_per_iter, theta_0, theta_star)
Utility.plot_norm_error(FDSA_theta_per_iter_norm_error, "FDSA theta")

FDSA_loss_per_iter = FDSA_optimizer.get_loss_per_iter()
FDSA_loss_per_iter_norm_error = Utility.get_loss_norm_error(FDSA_loss_per_iter, loss_0, loss_star)
Utility.plot_norm_error(FDSA_loss_per_iter_norm_error, "FDSA loss")

print("FDSA terminal loss error:", FDSA_loss_per_iter_norm_error[-1])

### CS-FDSA ###
print("running CS-FDSA")
CS_FDSA_optimizer = CsFDSA(a=a, A=A, alpha=alpha,
                           c=c, gamma=gamma,
                           iter_num=int(iter_num/p), rep_num=rep_num,
                           theta_0=theta_0, loss_obj=loss_obj,
                           record_theta_flag=True, record_loss_flag=True)
CS_FDSA_optimizer.train()

CS_FDSA_theta_per_iter = CS_FDSA_optimizer.get_theta_per_iter()
CS_FDSA_theta_per_iter_norm_error = Utility.get_theta_norm_error(CS_FDSA_theta_per_iter, theta_0, theta_star)
Utility.plot_norm_error(CS_FDSA_theta_per_iter_norm_error, "CS-FDSA theta")

CS_FDSA_loss_per_iter = CS_FDSA_optimizer.get_loss_per_iter()
CS_FDSA_loss_per_iter_norm_error = Utility.get_loss_norm_error(CS_FDSA_loss_per_iter, loss_0, loss_star)
Utility.plot_norm_error(CS_FDSA_loss_per_iter_norm_error, "CS-FDSA loss")

print("CS-FDSA terminal loss error:", CS_FDSA_loss_per_iter_norm_error[-1])

### SPSA ###
print("running SPSA")
SPSA_optimizer = SPSA(a=a, A=A, alpha=alpha,
                      c=c, gamma=gamma,
                      iter_num=int(iter_num/2), rep_num=rep_num,
                      theta_0=theta_0, loss_obj=loss_obj,
                      record_theta_flag=True, record_loss_flag=True)
SPSA_optimizer.train()

SPSA_theta_per_iter = SPSA_optimizer.get_theta_per_iter()
SPSA_theta_per_iter_norm_error = Utility.get_theta_norm_error(SPSA_theta_per_iter, theta_0, theta_star)
Utility.plot_norm_error(SPSA_theta_per_iter_norm_error, "SPSA theta")

SPSA_loss_per_iter = SPSA_optimizer.get_loss_per_iter()
SPSA_loss_per_iter_norm_error = Utility.get_loss_norm_error(SPSA_loss_per_iter, loss_0, loss_star)
Utility.plot_norm_error(SPSA_loss_per_iter_norm_error, "SPSA loss")

print("SPSA terminal loss error:", SPSA_loss_per_iter_norm_error[-1])

### CS-SPSA ###
print("running CS-SPSA")
CS_SPSA_optimizer = CsSPSA(a=a, A=A, alpha=alpha,
                           c=c, gamma=gamma,
                           iter_num=iter_num, rep_num=rep_num,
                           theta_0=theta_0, loss_obj=loss_obj,
                           record_theta_flag=True, record_loss_flag=True)
CS_SPSA_optimizer.train()

CS_SPSA_theta_per_iter = CS_SPSA_optimizer.get_theta_per_iter()
CS_SPSA_theta_per_iter_norm_error = Utility.get_theta_norm_error(CS_SPSA_theta_per_iter, theta_0, theta_star)
Utility.plot_norm_error(CS_SPSA_theta_per_iter_norm_error, "CS-SPSA theta")

CS_SPSA_loss_per_iter = CS_SPSA_optimizer.get_loss_per_iter()
CS_SPSA_loss_per_iter_norm_error = Utility.get_loss_norm_error(CS_SPSA_loss_per_iter, loss_0, loss_star)
Utility.plot_norm_error(CS_SPSA_loss_per_iter_norm_error, "CS-SPSA loss")

print("CS-SPSA terminal loss error:", CS_SPSA_loss_per_iter_norm_error[-1])

### plot ###
today = date.today()

# plot loss
fig = plt.figure()
ax = fig.add_subplot()
plt.grid()
line_FDSA, = ax.plot(np.concatenate(([1], np.repeat(FDSA_loss_per_iter_norm_error, int(2*p)))), 'k:')
line_CS_FDSA, = ax.plot(np.concatenate(([1], np.repeat(CS_FDSA_loss_per_iter_norm_error, int(p)))), 'k-.')
line_SPSA, = ax.plot(np.concatenate(([1], np.repeat(SPSA_loss_per_iter_norm_error, int(2)))), 'k--', dashes=(5,5))
line_SPSA_for_legend, = ax.plot([1], 'k--')
line_CS_SPSA, = ax.plot(np.concatenate(([1], np.repeat(CS_SPSA_loss_per_iter_norm_error, int(1)))), 'k-')

plt.xlim(xmin=0, xmax=iter_num)
plt.yscale("log")
plt.legend((line_FDSA, line_CS_FDSA, line_SPSA_for_legend, line_CS_SPSA),
           ("FDSA", "CS-FDSA", "SPSA", "CS-SPSA"), loc="best")
plt.xlabel("Number of Function Measurements")
plt.ylabel("Normalized Errors in Loss (log scale)")
plt.savefig("figure/quadatic_exponential_p_" + str(p) + "_loss_" + str(today) + ".pdf")
plt.show()

# plot theta
fig = plt.figure()
ax = fig.add_subplot()
plt.grid()
line_FDSA, = ax.plot(np.concatenate(([1], np.repeat(FDSA_theta_per_iter_norm_error, int(2*p)))), 'k:')
line_CS_FDSA, = ax.plot(np.concatenate(([1], np.repeat(CS_FDSA_theta_per_iter_norm_error, int(p)))), 'k-.')
line_SPSA, = ax.plot(np.concatenate(([1], np.repeat(SPSA_theta_per_iter_norm_error, int(2)))), 'k--', dashes=(10,10))
line_SPSA_for_legend, = ax.plot([1], 'k--')
line_CS_SPSA, = ax.plot(np.concatenate(([1], np.repeat(CS_SPSA_theta_per_iter_norm_error, int(1)))), 'k-')

plt.xlim(xmin=0, xmax=iter_num)
plt.yscale("log")
plt.legend((line_FDSA, line_CS_FDSA, line_SPSA_for_legend, line_CS_SPSA),
           ("FDSA", "CS-FDSA", "SPSA", "CS-SPSA"), loc="best")
plt.xlabel("Number of Function Measurements")
plt.ylabel(r"Normalized Errors in Estimation of $\mathbf{\theta}$ (log scale)")
plt.savefig("figure/quadatic_exponential_p_" + str(p) + "_theta_" + str(today) + ".pdf")
plt.show()

### APL plot ###
plt.figure()
plt.grid()
plt.plot(np.concatenate(([1], np.repeat(FDSA_loss_per_iter_norm_error, int(2*p)))))
plt.plot(np.concatenate(([1], np.repeat(SPSA_loss_per_iter_norm_error, int(2)))))
plt.plot(np.concatenate(([1], np.repeat(CS_SPSA_loss_per_iter_norm_error, int(1)))))

plt.xlim(xmin=0, xmax=iter_num)
plt.yscale("log")
plt.legend(("FDSA", "SPSA", "CS-SPSA"), loc="best")
plt.xlabel("Number of Function Measurements")
plt.ylabel("Normalized Errors in Loss (log scale)")
plt.savefig("figure/quadatic_exponential_p_" + str(p) + "_loss_" + str(today) + "_APL.pdf")
plt.show()

print("finish plotting")