# -*- coding: utf-8 -*-
"""
Simple scalar interpolation examples
"""

import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.path.relpath("../../src"))
import hiped as hp

#%% 1) Domain definition

domain = hp.Domain(5)

plt.figure()
domain.plot()
plt.show()

#%% 2) Definition of scalar functions at each vertex

f1s = hp.VertexFunction(label = "f1(u)", f = lambda u : u**2, dfdu = lambda u : 2*u)
f2s = hp.VertexFunction("f2(u)", lambda u : u**3, lambda u : 3*u**2)
f3s = hp.VertexFunction("f3(u)", lambda u : 0.1*u, lambda u : 0.1*np.ones(u.shape))
f4s = hp.VertexFunction("f4(u)", lambda u : np.ones(u.shape), lambda u : np.zeros(u.shape))
f5s = hp.VertexFunction("f5(u)", lambda u : 1/u, lambda u : -1/u**2)

plt.figure()
f1s.plot()
plt.show()

#%% 3) Definition of the interpolation

penal = hp.Penalization("simp", coeffPenal = 2, reverse=False) # "simp", "ramp", "lukas", "zhou"
plt.figure()
penal.plot()
plt.show()

interpScalar= hp.Interpolation(domain = domain, children = [f1s,f2s,f3s,f4s,f5s],
                               label = "interpScalar", penalization = penal)

#%% 4) Coordinates x in the polygon and scalar field u

n = 100
x = interpScalar.setInitialVariable(n, typeInit = "rand", radius = 1) # initialization of the variables
x = interpScalar.projection(x) # projection onto the domain
u = np.random.rand(1,1,n)

plt.figure()
interpScalar.plot(x)
plt.show()

#%% 5) Evaluation of the interpolation
#%% a) Interpolated values

import time
nRuns = 1000
t0 = time.time()
for i in range(nRuns):
    f = interpScalar.eval(x, u) # evaluate the interpolation
tv1 = time.time() - t0
print(f"Compute interpolated values in {tv1*1000:.3f} ms ({u.shape[2]} values, {nRuns} runs)")

#%% b) Derivative with respect to the scalar field u

nRuns = 1000
t0 = time.time()
for i in range(nRuns):
    dfdu =  interpScalar.evaldu(x, u) # evaluate the derivative of the interpolation w.r.t u
tv1 = time.time() - t0
print(f"Compute u-derivative in {tv1*1000:.3f} ms ({u.shape[2]} values, {nRuns} runs)")

#%% Check the accuracy of the Taylor expansion

from hiped.utils import mult
h = np.logspace(-8,-2,10) # test for 10 different h
resU = np.zeros((10,1,u.shape[2]))

for i in range(10):
    pert = h[i]*np.random.rand(*u.shape)
    fPerturbedu = interpScalar.eval(x,u+pert)
    resU[i,0,:] = np.abs(np.linalg.norm(fPerturbedu - (f + mult(dfdu,pert)), axis = 0))
    
maxResU = np.max(resU, axis = 2); medResU = np.median(resU, axis = 2);

plt.figure()
plt.loglog(h, medResU,'-o', label =  "median of residual")
plt.loglog(h, maxResU,'-o', label =  "maximum of residual")
plt.loglog(h, h**2,'k--', label =  "expected decay $(h^2)$")
plt.legend(loc = "best"); plt.grid()
plt.xlabel("$h$"); plt.ylabel("Euclidian norm of Taylor remainder")
plt.title("Taylor remainder with respect to $u$")
plt.show()

#%% c) Derivative with respect to the cartesian coordinates x in the polygon

t0 = time.time()
for i in range(nRuns):
    dfdx =  interpScalar.evaldx(x, u) # evaluate the derivative of the interpolation w.r.t x
tv1 = time.time() - t0
print(f"Compute x-derivative in {tv1*1000:.3f} ms ({u.shape[2]} values, {nRuns} runs)")

#%% Check the accuracy of the Taylor expansion

resX = np.zeros((10,1,u.shape[2]))
l = list(x.keys())[0]
for i in range(10):
    xPert = x.copy()
    pert = h[i]*np.random.rand(*x[l].shape)
    xPert[l] = x[l] + pert
    xPert = interpScalar.projection(xPert)
    pert = xPert[l] - x[l]
    fPerturbedx = interpScalar.eval(xPert,u)
    pert = np.reshape(pert.T, (x[l].shape[1],1,-1))
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

#%% d) Speed tip

'''
To compute several times the interpolation for different $u$  without changing 
the position x in the domain (for example, when solving a non-linear system 
in u), one can first compute the shape functions once for all. '''

u2 = np.random.rand(1,nRuns,u.shape[2])

# naive
t0 = time.time()
for i in range(nRuns):
    interpScalar.eval(x, u2[:,i,:]) # value
    interpScalar.evaldu(x, u2[:,i,:]) # derivative w.r.t u
    interpScalar.evaldx(x, u2[:,i,:]) # derivative w.r.t x
tNaive = time.time() - t0
print(f"Computation time (naive version, {nRuns} runs) : {tNaive*1000:.3f} ms")

# faster
t0 = time.time()
# pre-computation of the shape functions that don't depend on u
w, dwdx = interpScalar.evalBasisFunction(x)
for i in range(nRuns):
    interpScalar.eval(x, u2[:,i,:], w) # value
    interpScalar.evaldu(x, u2[:,i,:], w) # derivative w.r.t u
    interpScalar.evaldx(x, u2[:,i,:], w, dwdx) # derivative w.r.t x
tOptim = time.time() - t0
print(f"Computation time (optimized version, {nRuns} runs) : {tOptim*1000:.3f} ms")