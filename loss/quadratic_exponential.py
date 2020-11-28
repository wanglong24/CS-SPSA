#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:57:59 2020

@author: longwang
"""

import numpy as np
from scipy.optimize import minimize

class QuadraticExponential:
    def __init__(self, p):
        np.random.seed(99)

        self.p = p
        if p == 10:
            self.eta = np.array([1.1025, 1.6945, 1.4789, 1.9262, 0.7505,
                                 1.3267, 0.8428, 0.7247, 0.7693, 1.3986])
            # self.theta_star = np.array([0.286, 0.229, 0.248, 0.211, 0.325,
            #                             0.263, 0.315, 0.327, 0.323, 0.256])
        else:
            self.eta = np.random.rand(self.p) * 1.5 + 0.5

    def get_theta_star(self):
        theta_0 = np.ones(self.p) * 0.5
        res = minimize(self.get_loss_true, theta_0, method='BFGS', tol=1e-6)
        return res.x

    def get_loss_noisy(self, iter_idx, theta):
        X = np.random.exponential(1 / self.eta)
        return np.sum(theta ** 2) + np.sum(np.exp(-(X * theta)))

    def get_loss_noisy_complex(self, iter_idx, theta):
        X = np.random.exponential(1 / self.eta)
        return np.sum(theta ** 2) + np.sum(np.exp(-(X * theta)))

    def get_loss_true(self, theta):
        return np.sum(theta ** 2) + np.sum(self.eta / (self.eta + theta))

    def get_grad_true(self, theta):
        return 2 * theta - self.eta / (self.eta + theta) ** 2

if __name__ == "__main__":
    p = 10
    loss_obj = QuadraticExponential(p)
    theta_0 = np.ones(p)
    loss_noisy = loss_obj.get_loss_noisy(0, theta_0)
    loss_true = loss_obj.get_loss_true(theta_0)
    print(loss_noisy, loss_true)


    theta_star = loss_obj.get_theta_star()
    print(theta_star)