#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Long Wang
"""

import numpy as np
import matplotlib.pyplot as plt

class Utility(object):
    def get_theta_norm_error(theta, theta_0, theta_star):
        return np.linalg.norm(theta - theta_star, axis=1) / np.linalg.norm(theta_0 - theta_star)

    def get_loss_norm_error(loss, loss_0, loss_star):
        return (loss - loss_star) / (loss_0 - loss_star)

    def plot(norm_error, algo_name, log_scale=True):
        iter_num = norm_error.size

        plt.figure()
        plt.plot(np.concatenate(([1], norm_error)))
        plt.xlim(xmin=0, xmax=iter_num)
        if log_scale:
            plt.yscale("log")
        plt.title(algo_name)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Normalized Error")
        plt.show()