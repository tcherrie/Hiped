# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:12:32 2024

@author: tcherriere
"""

import numpy as np
from ngsolve import GridFunction

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

def array2gf(array, space):
    s = array.shape
    gf = []
    if len(s)==1:
        gf.append(GridFunction(space))
        gf[0].vec.FV().NumPy()[:] = array
    elif len(s)==2:
        for i in range(s[1]):
            gf.append(GridFunction(space))
            gf[i].vec.FV().NumPy()[:] = array[:,i]
    elif len(s)==3: # w and dwdx
        for i in range(s[0]):
            gfint = []
            for j in range(s[1]):
                gfint.append(GridFunction(space))
                gfint[j].vec.FV().NumPy()[:] = array[i,j,:]
            gf.append(gfint)
                
    return gf

def gf2array(gfList):
    Array = []
    dim = len(gfList)
    if isinstance(gfList[0], GridFunction):
        space = gfList[0].space
        for gf in gfList:
            Array.append(gf.vec.FV().NumPy()[:])
    else :
        a, space = gf2array(gf)
        Array.append(a)
    Array = np.array(Array)
    if len(Array.shape)==2:
        Array = Array.T
    elif len(Array.shape)==3:
        Array = Array.transpose(0,2,1)
    return Array, space