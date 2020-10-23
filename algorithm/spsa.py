#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:31:45 2020

@author: Long Wang
"""

import numpy as np
from algorithm.gradient_descent_algo import GradDescAlgo

class SPSA(GradDescAlgo):
    def get_delta_all(self):
        self.delta_all = np.round(np.random.rand(self.p, self.direct_num, self.iter_num, self.rep_num)) * 2 - 1

    def get_grad_est(self, iter_idx=0, rep_idx=0, theta_k=None):
        grad_k_all = np.empty((self.p, self.direct_num))

        c_k = self.c / (iter_idx + 1) ** self.gamma
        for direct_idx in range(self.direct_num):
            delta_k = self.delta_all[:,direct_idx, iter_idx, rep_idx]
            loss_plus = self.loss_noisy(theta_k + c_k * delta_k)
            loss_minus = self.loss_noisy(theta_k - c_k * delta_k)
            grad_k_all[:,direct_idx] = (loss_plus - loss_minus) / (2 * c_k * delta_k)

        return np.average(grad_k_all, axis=1)

    def train(self):
        if self.record_theta_flag:
            self.theta_k_all = np.empty((self.p, self.iter_num, self.rep_num))
        if self.record_loss_flag:
            self.loss_k_all = np.empty((self.iter_num, self.rep_num))

        self.get_delta_all()

        for rep_idx in range(self.rep_num):
            print("running rep_idx:", rep_idx+1, "/", self.rep_num)
            # reset theta
            theta_k = self.theta_0.copy()
            for iter_idx in range(self.iter_num):
                a_k = self.a / (iter_idx + 1 + self.A) ** self.alpha
                g_k = self.get_grad_est(iter_idx, rep_idx, theta_k)
                theta_k -= a_k * g_k

                # record result
                if self.record_theta_flag:
                    self.theta_k_all[:,iter_idx,rep_idx] = theta_k
                if self.record_loss_flag:
                    self.loss_k_all[iter_idx,rep_idx] = self.loss_true(theta_k)