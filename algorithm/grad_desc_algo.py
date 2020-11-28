#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 23:41:54 2020

@author: Long Wang
"""

import numpy as np

class GradDescAlgo(object):

    def __init__(self,
                 a=0, A=0, alpha=0.602,
                 iter_num=1, rep_num=1, direct_num=1,
                 theta_0=None, loss_obj=None,
                 record_theta_flag=False, record_loss_flag=False):

        np.random.seed(99)

        # step size: a_k = a / (k+1+A) ** alpha
        self.a = a
        self.A = A
        self.alpha = alpha

        self.iter_num = iter_num
        self.rep_num = rep_num
        self.direct_num = direct_num

        self.theta_0 = theta_0
        self.loss_obj = loss_obj

        self.record_theta_flag = record_theta_flag
        self.record_loss_flag = record_loss_flag

        # induced parameter
        self.p = theta_0.size

    def get_grad_est(self, iter_idx, theta_k):
        pass

    def train(self):
        if self.record_theta_flag:
            self.theta_ks = np.empty((self.rep_num, self.iter_num, self.p))
        if self.record_loss_flag:
            self.loss_ks = np.empty((self.rep_num, self.iter_num))

        for rep_idx in range(self.rep_num):
            print("running rep_idx:", rep_idx + 1, "/", self.rep_num)
            # reset theta
            theta_k = self.theta_0.copy()
            for iter_idx in range(self.iter_num):
                a_k = self.a / (iter_idx + 1 + self.A) ** self.alpha
                g_k = self.get_grad_est(iter_idx, theta_k)
                theta_k -= a_k * g_k

                # record result
                if self.record_theta_flag:
                    self.theta_ks[rep_idx,iter_idx] = theta_k
                if self.record_loss_flag:
                    self.loss_ks[rep_idx,iter_idx] = self.loss_obj.get_loss_true(theta_k)

    def get_theta_per_iter(self):
        return np.mean(self.theta_ks, axis=0) # average over rep_num

    def get_loss_per_iter(self):
        return np.mean(self.loss_ks, axis=0) # average over rep_num