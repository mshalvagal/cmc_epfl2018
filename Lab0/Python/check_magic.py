# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:55:02 2018

@author: manu
"""

import numpy as np

def check_magic_fun(M):
    col_sums = np.sum(M,axis=0)
    if max(col_sums)!=min(col_sums):
        return False
    else:
        value = max(col_sums)
    row_sums = np.sum(M,axis=1)
    if max(row_sums)!=min(row_sums) or max(row_sums)!=value:
        return False
    if M.trace()!=value or np.fliplr(M).trace()!=value:
        return False
    return True