#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 23:41:54 2020

@author: Long Wang
"""

class GradDescAlgo(object):

    def __init__(self, a=0, c=0, A=0, alpha=0.602, gamma=0.101,
                 iter_num=1, rep_num=1, direct_num=1,
                 theta_0=None, loss_true=None, loss_noisy=None,
                 record_theta_flag=True, record_loss_flag=True):

        # step size: a_k = a / (k+1+A) ** alpha
        # perturbation size: c_k = c / (k+1) ** gamma
        # direct_num: number of directions per iteration

        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma

        self.iter_num = iter_num
        self.rep_num = rep_num
        self.direct_num = direct_num

        self.theta_0 = theta_0
        self.loss_true = loss_true
        self.loss_noisy = loss_noisy

        self.record_theta_flag = record_theta_flag
        self.record_loss_flag = record_loss_flag

        self.p = theta_0.shape[0]

    def get_grad_est(self):
        pass

    def train(self):
        pass