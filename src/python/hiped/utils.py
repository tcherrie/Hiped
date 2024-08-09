# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:12:32 2024

@author: tcherriere
"""

import numpy as np

def t(a):
    """
    Returns the page-wise transpose of a (shortcut for np.swapaxes(a,1,0))
    
    Parameters:
    ----------
    a : nd array
        
    Returns:
    -------
    nd array
        The page-wise transposed of a
    """
    return np.swapaxes(a,1,0)


def mult(*args):
    """
    Returns the page-wise matrix multiplication of *args.
    
    Parameters:
    ----------
    *args : nd array
        
    Returns:
    -------
    nd array
        The page-wise multiplication of *args
    """
    result = args[0]
    for a in args[1:]: 
        result = np.einsum('ij...,jk...->ik...', result, a)
    return result