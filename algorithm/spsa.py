#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:31:45 2020

@author: Long Wang
"""

import numpy as np
from algorithm.grad_desc_algo import GradDescAlgo

class SPSA(GradDescAlgo):
    def __init__(self, c=0, gamma=0.101, **kwargs):
        super(SPSA, self).__init__(**kwargs)
        self.c = c
        self.gamma = gamma

    def get_grad_est(self, iter_idx, theta_k):
        grad_ks = np.empty((self.direct_num, self.p))

        c_k = self.c / (iter_idx + 1) ** self.gamma
        delta_ks = np.round(np.random.rand(self.p, self.direct_num)) * 2 - 1
        for direct_idx in range(self.direct_num):
            delta_k = delta_ks[:,direct_idx]
            loss_plus = self.loss_obj.get_loss_noisy(iter_idx, theta_k + c_k * delta_k)
            loss_minus = self.loss_obj.get_loss_noisy(iter_idx, theta_k - c_k * delta_k)
            grad_ks[direct_idx] = (loss_plus - loss_minus) / (2 * c_k * delta_k)

        return np.average(grad_ks, axis=0) # average over direct_num