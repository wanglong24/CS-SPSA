#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:35:30 2020

@author: longwang
"""

import numpy as np
from algorithm.grad_desc_algo import GradDescAlgo

class CsSPSA(GradDescAlgo):
    def __init__(self, c=0, gamma=0.101, **kwargs):
        super(CsSPSA, self).__init__(**kwargs)
        self.c = c
        self.gamma = gamma

    def get_grad_est(self, iter_idx, theta_k):
        grad_ks = np.empty((self.direct_num, self.p))

        c_k = self.c / (iter_idx + 1) ** self.gamma
        delta_ks = np.round(np.random.rand(self.p, self.direct_num)) * 2 - 1
        for direct_idx in range(self.direct_num):
            delta_k = delta_ks[:,direct_idx]
            theta_k_plus = np.array(theta_k, dtype = complex) + 1j * c_k * delta_k
            loss_plus = self.loss_obj.get_loss_noisy_complex(iter_idx, theta_k_plus)
            grad_ks[direct_idx] = loss_plus.imag / (c_k * delta_k)

        return np.average(grad_ks, axis=0)