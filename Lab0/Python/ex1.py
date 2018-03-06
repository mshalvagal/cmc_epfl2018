# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:18:33 2018

@author: manu
"""

import numpy as np
import check_magic as cm

print(cm.check_magic_fun(np.eye(3)))
print(cm.check_magic_fun(np.ones([3,3])))