#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:35:30 2020

@author: longwang
"""

import numpy as np
from algorithm.spsa import SPSA

class CsSPSA(SPSA):
    def get_grad_est(self, iter_idx=0, rep_idx=0, theta_k=None):
        grad_k_all = np.empty((self.p, self.direct_num))

        c_k = self.c / (iter_idx + 1) ** self.gamma
        for direct_idx in range(self.direct_num):
            delta_k = self.delta_all[:,direct_idx, iter_idx, rep_idx]
            theta_k_plus = np.array(theta_k, dtype = complex) + 1j * c_k * delta_k
            loss_plus = self.loss_noisy(theta_k_plus)
            grad_k_all[:,direct_idx] = loss_plus.imag / (c_k * delta_k)

        return np.average(grad_k_all, axis=1)