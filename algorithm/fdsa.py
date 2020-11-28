#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 23:14:32 2020

@author: Long Wang
"""

import numpy as np
from algorithm.grad_desc_algo import GradDescAlgo

class FDSA(GradDescAlgo):
    def __init__(self, c=0, gamma=0.101, **kwargs):
        super(FDSA, self).__init__(**kwargs)
        self.c = c
        self.gamma = gamma

    def get_grad_est(self, iter_idx, theta_k):
        grad_ks = np.empty((self.direct_num, self.p))

        c_k = self.c / (iter_idx + 1) ** self.gamma
        for direct_idx in range(self.direct_num):
            for i in range(self.p):
                theta_k_plus = theta_k.copy()
                theta_k_plus[i] += c_k
                theta_k_minus = theta_k.copy()
                theta_k_minus[i] -= c_k
                loss_plus = self.loss_obj.get_loss_noisy(iter_idx, theta_k_plus)
                loss_minus = self.loss_obj.get_loss_noisy(iter_idx, theta_k_minus)
                grad_ks[direct_idx,i] = (loss_plus - loss_minus) / (2 * c_k)

        return np.average(grad_ks, axis=0)