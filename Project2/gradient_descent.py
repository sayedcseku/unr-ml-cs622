# -*- coding: utf-8 -*-
"""
@author: Md Abu Sayed
"""
import numpy as np

def gradient_descent(grad_f, x_init, eta):
    x = x_init
    del_f = grad_f(x)
    #x = x - eta * del_f
    
    while np.linalg.norm(del_f) > 0.0001:
        x = x - eta * del_f
        del_f = grad_f(x)
    return x