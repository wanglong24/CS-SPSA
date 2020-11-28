#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 23:26:13 2020

@author: longwang
"""

import numpy as np
from algorithm.grad_desc_algo import GradDescAlgo

class CsFDSA(GradDescAlgo):
    def __init__(self, c=0, gamma=0.101, **kwargs):
        super(CsFDSA, self).__init__(**kwargs)
        self.c = c
        self.gamma = gamma

    def get_grad_est(self, iter_idx, theta_k):
        grad_ks = np.empty((self.direct_num, self.p))

        c_k = self.c / (iter_idx + 1) ** self.gamma
        for direct_idx in range(self.direct_num):
            for i in range(self.p):
                theta_k_plus = np.array(theta_k, dtype = complex)
                theta_k_plus[i] += 1j * c_k
                loss_plus = self.loss_obj.get_loss_noisy_complex(iter_idx, theta_k_plus)
                grad_ks[direct_idx] = loss_plus.imag / c_k

        return np.average(grad_ks, axis=0)