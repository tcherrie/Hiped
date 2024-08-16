# -*- coding: utf-8 -*-
"""
Operations on VertexFunctions
"""

import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.path.relpath("..//..//..//src//python"))
import hiped as hp

#%% 1) Scalar case
'''
Elementary VertexFunctions can be combined with others, the computation of the 
expression and derivative is done automatically.
'''
#%% a) Some operations

f1s = hp.VertexFunction("f1s", lambda u: u**2, lambda u: 2*u)
f2s = hp.VertexFunction("f2s", lambda u: u**3, lambda u: 3*u**2)
f3s = hp.VertexFunction("f3s", lambda u: 0.1*u, lambda u: 0.1*np.ones(u.shape))

# Operations with scalars

f4s = (2*f1s + 3)**2
print(f4s)

# Operations with other VertexFunctions

f5s = f1s*(f2s/3+f3s*2)**(-f2s)
print(f5s)
f6s = f1s/f3s
print(f6s)

#%% b) Interpolation

n = 100
penal = hp.Penalization("ramp", 3)
interpScalar= hp.Interpolation(domain = hp.Domain(6), children = [f1s,f2s,f3s,f4s,f5s,f6s],
                               label = "interpScalar", penalization = penal)


x = interpScalar.setInitialVariable(n, "rand", radius = 1) # initialization of the variables
x = interpScalar.projection(x) # projection onto the domain
u = np.random.rand(1,1,n)
plt.figure()
interpScalar.plot(x)
plt.show()

#%% c) Evaluation

w, dwdx = interpScalar.evalBasisFunction(x)
f = interpScalar.eval(x, u, w) # value
dfdu =  interpScalar.evaldu(x, u, w) # derivative w.r.t a
dfdx =  interpScalar.evaldx(x, u, w, dwdx) # derivative w.r.t x

#%% Check Taylor expansion - u

from hiped.utils import mult

# derivative with respect to u
h = np.logspace(-8,-2,10) # test for 10 different h
resU = np.zeros((10,1,u.shape[2]))

for i in range(10):
    pert = h[i]*np.random.rand(*u.shape)
    fPerturbedu = interpScalar.eval(x,u+pert)
    resU[i,0,:] = np.linalg.norm(fPerturbedu - (f + mult(dfdu,pert)), axis = 0)
    
maxResU = np.max(resU, axis = 2); medResU = np.median(resU, axis = 2);

plt.figure()
plt.loglog(h, medResU,'-o', label =  "median of the residual")
plt.loglog(h, maxResU,'-o', label =  "maximum of the residual")
plt.loglog(h, h**2,'k--', label =  "expected decay $(h^2)$")
plt.legend(loc = "best"); plt.grid()
plt.xlabel("$h$"); plt.ylabel("Euclidian norm of Taylor remainder")
plt.title("Taylor remainder with respect to $u$")
plt.show()

#%% Check Taylor expansion - x

resX = np.zeros((10,1,u.shape[2]))
l = list(x.keys())[0]
xPert = x.copy()

for i in range(10):
    pert = h[i]*np.random.rand(*x[l].shape)
    xPert[l] = x[l] + pert
    fPerturbedx = interpScalar.eval(xPert,u)
    pert = np.reshape(pert.T, (2,1,-1))
    resX[i,0,:] = np.linalg.norm(fPerturbedx - (f + mult(dfdx[l],pert)), axis = 0)

maxResX = np.max(resX, axis = 2); medResX = np.median(resX, axis = 2);

plt.figure()
plt.loglog(h, medResX,'-o', label =  "median of the residual")
plt.loglog(h, maxResX,'-o', label =  "maximum of the residual")
plt.loglog(h, h**2,'k--', label =  "expected decay $(h^2)$")
plt.legend(loc = "best"); plt.grid()
plt.xlabel("$h$"); plt.ylabel("Euclidian norm of Taylor remainder")
plt.title("Taylor remainder with respect to $x$")
plt.show()

#%% 2) Vector case
#%% a) Some operations

from hiped.utils import mult

dimInput = 2
dimOutput = 2

f1v = hp.VertexFunction(label = "f1v", f = lambda u : u**2, dfdu = lambda u : 2*u * np.eye(2)[:,:,None], dimInput= dimInput, dimOutput = dimOutput)
f2v = hp.VertexFunction("f2v", lambda u : mult(np.array([[1,2],[3,4]]), u), lambda u : np.array([[1,2],[3,4]])[:,:,None] * np.ones(u.shape), dimInput, dimOutput)
f3v = hp.VertexFunction("f3v", lambda u : 0.1*u, lambda u : 0.1 * np.eye(2)[:,:,None] * np.ones(u.shape), dimInput, dimOutput)

fMat =  hp.VertexFunction("fMat", lambda u : np.array([[1,2],[3,4]]).reshape(2,2,1)*np.ones(u.shape), lambda u : np.zeros(u.shape) * np.zeros((2,2,1)), dimInput, (2,2))
fConst =  hp.VertexFunction("fConst", lambda u : np.ones((1,1,1))*np.ones(u.shape), lambda u : np.zeros((1,1,1))*np.zeros(u.shape), dimInput, 1)
fConst2 =  hp.VertexFunction("fConst2", lambda u : np.ones((2,1,1))*np.ones(u.shape), lambda u : np.zeros((2,2,1))*np.zeros(u.shape), dimInput, 2)

f4v = fMat @ f1v * (f3v+f2v) / 2 # only multiplication by constant matrix is supported
print(f4v)
f5v = 3* f1v.innerProduct(f2v+f3v)**f1v 
print(f5v)

#%% b) Interpolation

n = 100
penal = hp.Penalization("simp", 2)
interpVector= hp.Interpolation(domain = hp.Domain(5), children = [f1v,f2v,f3v,f4v,f5v],
                               label = "interpVector", penalization = penal)


x = interpVector.setInitialVariable(n, "rand", radius = 1) # initialization of the variables
x = interpVector.projection(x) # projection onto the domain
u = np.random.rand(dimInput,1,n)
plt.figure()
interpVector.plot(x)
plt.show()

#%% c) Evaluation

w, dwdx = interpVector.evalBasisFunction(x)
f = interpVector.eval(x, u, w) # value
dfdu =  interpVector.evaldu(x, u, w) # derivative w.r.t a
dfdx =  interpVector.evaldx(x, u, w, dwdx) # derivative w.r.t x

#%% Check Taylor expansion - u

h = np.logspace(-8,-2,10) # test for 10 different h
resU = np.zeros((10,1,u.shape[2]))

for i in range(10):
    pert = h[i]*np.random.rand(*u.shape)
    fPerturbedu = interpVector.eval(x,u+pert)
    resU[i,0,:] = np.linalg.norm(fPerturbedu - (f + mult(dfdu,pert)), axis = 0)
    
maxResU = np.max(resU, axis = 2); medResU = np.median(resU, axis = 2);

plt.figure()
plt.loglog(h, medResU,'-o', label =  "median of the residual")
plt.loglog(h, maxResU,'-o', label =  "maximum of the residual")
plt.loglog(h, h**2,'k--', label =  "expected decay $(h^2)$")
plt.legend(loc = "best"); plt.grid()
plt.xlabel("$h$"); plt.ylabel("Euclidian norm of Taylor remainder")
plt.title("Taylor remainder with respect to $u$")
plt.show()

#%% Check Taylor expansion - x

resX = np.zeros((10,1,u.shape[2]))
l = list(x.keys())[0]
xPert = x.copy()

for i in range(10):
    pert = h[i]*np.random.rand(*x[l].shape)
    xPert[l] = x[l] + pert
    fPerturbedx = interpVector.eval(xPert,u)
    pert = np.reshape(pert.T, (2,1,-1))
    resX[i,0,:] = np.linalg.norm(fPerturbedx - (f + mult(dfdx[l],pert)), axis = 0)

maxResX = np.max(resX, axis = 2); medResX = np.median(resX, axis = 2);

plt.figure()
plt.loglog(h, medResX,'-o', label =  "median of the residual")
plt.loglog(h, maxResX,'-o', label =  "maximum of the residual")
plt.loglog(h, h**2,'k--', label =  "expected decay $(h^2)$")
plt.legend(loc = "best"); plt.grid()
plt.xlabel("$h$"); plt.ylabel("Euclidian norm of Taylor remainder")
plt.title("Taylor remainder with respect to $x$")
plt.show()

