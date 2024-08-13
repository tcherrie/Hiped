# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 18:51:38 2024

@author: tcherriere
"""

from .utils import t, mult, array2gf, gf2array
from .vertexFunction import VertexFunction
from .domain import *
from .penalization import *
from .vizualization import *

from ngsolve import GridFunction, CoefficientFunction

import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.tri import Triangulation
import matplotlib.colors as mcolors
from scipy.optimize import minimize
from copy import deepcopy, copy

class Interpolation:
    '''
    Interpolation Class
    
     To create an Interpolation object, type
     obj = Interpolation(structure)
    
     where 'structure' contains the following fields :
     - 'Domain' : an instance of the Domain class with n vertices
     - 'Children' : a vector of n VertexFunction or Interpolation instances
     - 'Label' : a unique string identifier
     - 'Penalization' (facultative) : an array of Penalization instances
    
     An Interpolation object describes a tree. The leaves are
     VertexFunctions, interpolated by ShapeFunctios and Penalizations. See
     the examples files for some illustrations.
    
    Copyright (C) 2024 Théodore CHERRIERE (theodore.cherriere@ricam.oeaw.ac.at.fr)
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    '''
    def __init__(self, domain, children, label, penalization = [], typeShapeFunction = "wachspress",
                 dimInput = None, dimOutput = None):
        '''
        obj = Interpolation(structure)
        where 'structure' contains the following fields :
        - 'Domain' : an instance of the Domain class with n vertices
        - 'Children' : a vector of n VertexFunction or Interpolation instances
        - 'Label' : a unique string identifier
        - 'Penalization' (facultative) : an array of Penalization instances 
        %
        see "b_SimpleScalarInterp", "c_SimpleVectorInterp", 
        "d_HierarchicalScalarInterp" and "e_HierarchicalVectorInterp"
        for some examples.
        '''
        
        # Checks
        assert isinstance(domain, Domain), 'domain should be part of the Domain class' 
        
        for child in children:
            pass
            # assert isinstance(child, Interpolation) or isinstance(child, VertexFunction), "children's  should be part of the Interpolation or VertexFunction classes" 
        
        if not isinstance(penalization,Penalization):
            for penal in penalization:
                assert isinstance(penal, Penalization), "penalization's elements should be part of Penalization class"
        
        nv = domain.Vertices.shape[0]
        assert len(children)==nv, "The number of children does not match the number of vertices"


        self.Children = children
        #self.DimInput = self.Children[0].DimInput
        #self.DimOutput = self.Children[0].DimOutput
        self.Label = label

        self.ShapeFunction = ShapeFunction(domain = domain, label = label, type = typeShapeFunction)
        
        self.DimInput = dimInput
        self.DimOutput = dimOutput
        
        if self.DimInput is None : self.DimInput = children[0].DimInput
        if self.DimOutput is None : self.DimOutput = children[0].DimOutput

        
        if isinstance(penalization, Penalization) :
            self.Penalization = [deepcopy(penalization) for i in range(nv)]
        elif len(penalization) == 0:
            self.Penalization = [Penalization("simp") for i in range(nv)]
        elif len(penalization) == 1:
            self.Penalization = [penalization for i in range(nv)]
        elif len(penalization) == nv :
            self.Penalization = penalization
        else:
            raise Exception("The number of penalization's elements does not match the number of domain's vertices")
    
    def setInitialVariable(self, nVariables = 10, typeInit = "zero", radius = 1, x00 = dict(), NGSpace = None):
        if x00 == dict(): x0 = x00.copy() 
        else: x0 = x00
        assert self.Label not in x0.keys(), "Non-unique interpolation labels"
        dim = self.ShapeFunction.Domain.Dimension
        
        if NGSpace is not None: nVariables = len(GridFunction(NGSpace).vec.FV())
        
        if typeInit.lower() in ["zero", "zeros", "z", "null", "center"]:
            x0[self.Label] = np.zeros((nVariables, dim))
            if NGSpace is not None:
                x0[self.Label] = [GridFunction(NGSpace) for i in range(dim)]
            
        elif typeInit.lower() in ["rand", "random", "r"]:
            if dim == 0:
                x0[self.Label] = np.zeros((nVariables, dim))
                if NGSpace is not None:
                    x0[self.Label] = [GridFunction(NGSpace)]
            elif dim == 1:
                x0[self.Label] = radius*(np.random.rand(nVariables, dim)-0.5)
                if NGSpace is not None:
                    x0[self.Label] = [GridFunction(NGSpace)]
                    x0[self.Label][0].vec.FV().NumPy()[:] = radius*(np.random.rand(nVariables, dim)-0.5)
            elif dim == 2:
                R = radius*np.random.rand(nVariables)
                th = np.random.rand(nVariables)*2*np.pi
                x, y = R * np.cos(th), R * np.sin(th)
                x0[self.Label] = np.array([x,y]).T
                if NGSpace is not None:
                    xyz = [x,y]
                    x0[self.Label] = [GridFunction(NGSpace) for i in range(dim)]
                    for i in range(len(x0[self.Label])):
                        x0[self.Label][i].vec.FV().NumPy()[:] = xyz[i]
            elif dim == 3:
                R = radius*np.random.rand(nVariables)
                th = np.random.rand(nVariables)*2*np.pi
                phi = np.random.rand(nVariables)*2*np.pi
                x = R * np.cos(th)* np.cos(phi)
                y = R * np.cos(th)* np.sin(phi)
                z = R * np.sin(th)
                x0[self.Label] = np.array([x,y,z]).T
                if NGSpace is not None:
                    xyz = [x,y,z]
                    x0[self.Label] = [GridFunction(NGSpace) for i in range(dim)]
                    for i in range(len(x0[self.Label])):
                        x0[self.Label][i].vec.FV().NumPy()[:] = xyz[i]
        for child in self.Children:
            if isinstance(child,Interpolation):
                x0 = child.setInitialVariable(nVariables, typeInit, radius, x0)
                
        return x0
    
    
    def projection(self, x, xproj = None, copyFlag = False, deepcopyFlag = False):
        if xproj is None :
            if deepcopyFlag : xproj = deepcopy(x)
            elif copyFlag : xproj = copy(x)
            else : xproj =x
        xproj[self.Label] = self.ShapeFunction.Domain.projection(x[self.Label])
        for child in self.Children:
            if isinstance(child, Interpolation):
                xproj = child.projection(x, xproj)
        return xproj
    
    ''' 
    Evaluation
    '''
    
    def evalBasisFunction(self, x, w=dict(), dwdx=dict()):
        val = x[self.Label]
        flagNGsolve = False
        if isinstance(val, list): #  transformation to NumPy array if NGsolve
            flagNGsolve = True
            val, space = gf2array(val)
            
        w[self.Label], dwdx[self.Label] = self.ShapeFunction.eval(val)
        
        if flagNGsolve: # transformation to GridFunction if NGSolve
            w[self.Label] = array2gf(w[self.Label], space)
            w[self.Label] = [l[0] for l in w[self.Label]]
            dwdx[self.Label] = array2gf(dwdx[self.Label], space)
            
        for child in self.Children:
            if isinstance(child, Interpolation):
                w, dwdx = child.evalBasisFunction(x, w, dwdx)
                
        return w, dwdx
    
    
    def eval(self, x, u, w_=dict(), flagCompil = True):
        # x = dict
        # u = dimInput x 1 x N
        
        if len(w_) == 0: w, _ = self.evalBasisFunction(x)
        else: w = w_.copy()
        
        if isinstance(x[self.Label], list): # NGsolve
            result = self.__eval_NGsolve(x, u, w, flagCompil)
        else:
        
            if len(u.shape) < 3 : u= u.T.reshape(self.DimInput,1,-1)
            
            sz = u.shape[2]
            nChildren = len(self.Children)
    
            # 1) computation of the children
            coeff = np.zeros((self.Children[0].DimOutput,1, sz, nChildren));
            
            for i in range(nChildren):
                child = self.Children[i]
                if isinstance(child, Interpolation):
                    coeff[:,:,:,i] = child.eval(x, u, w)
                else:
                    coeff[:,:,:,i] = child.eval(u)
        
            # 2) multiplication by the shape functions
            val = np.zeros((1,1,sz,nChildren))
            for i in range(nChildren):
                val[:,:,:,i] = self.Penalization[i].eval(w[self.Label][i,:,:])
            
            
            # 3) Result
            result = np.sum(coeff * val, 3)
        return result
    
    
    def __eval_NGsolve(self, x, u, w, flagCompil = True):
        nChildren = len(self.Children)
        
        # 1) computation of the children
        
        coeff = [None  for i in range(nChildren)]
        for i in range(nChildren):
            child = self.Children[i]
            if isinstance(child, Interpolation):
                coeff[i] = child.__eval_NGsolve(x, u, w)
            else:
                coeff[i] = child.eval(u)
        
        # 2) multiplication by the shape functions
        
        x, space = gf2array(w[self.Label])
        valGf = [GridFunction(space) for i in range(nChildren)]
        for i in range(nChildren):
            valGf[i].vec.FV().NumPy()[:] = self.Penalization[i].eval(x[:,i])
            
        
        # 3) Result
        result = CoefficientFunction(0)
        for i in range(nChildren):
            result = valGf[i]*coeff[i] + result
        if flagCompil: return result.Compile()
        else : return result

    def evaldu(self, x, u, w_=dict(), flagCompil = True):
        if len(w_) == 0: w, _ = self.evalBasisFunction(x)
        else: w = w_.copy()
        
        if isinstance(x[self.Label], list): # NGsolve
            result = self.__evaldu_NGsolve(x, u, w, flagCompil)
        else:
        
            if len(u.shape) < 3 : u = u= u.T.reshape(self.DimInput,1,-1)
            
            sz = u.shape[2]
            nChildren = len(self.Children)
            d1 = self.Children[0].DimOutput
            d2 = self.Children[0].DimInput
            
            # 1) computation of the derivative of the children
            coeffd = np.zeros((d1,d2, sz, nChildren))
            for i in range(nChildren):
                child = self.Children[i]
                if isinstance(child, Interpolation):
                    coeffd[:,:,:,i] = child.evaldu(x, u, w)
                else:
                    coeffd[:,:,:,i] = child.evald(u)
                    
            # 2) multiplication by the shape functions
            Pw=np.zeros((1,1,sz,nChildren))
            for i in range(nChildren):
                Pw[:,:,:,i]=self.Penalization[i].eval(w[self.Label][i,:,:])
            
            # 3) result
            result = np.sum(coeffd*Pw, 3)
        return result
    
    def __evaldu_NGsolve(self, x, u, w, flagCompil = True):
        nChildren = len(self.Children)

        # 1) computation of the children

        coeff = [None  for i in range(nChildren)]
        for i in range(nChildren):
            child = self.Children[i]
            if isinstance(child, Interpolation):
                coeff[i] = child.__evaldu_NGsolve(x, u, w)
            else:
                coeff[i] = child.evald(u)
        
        # 2) multiplication by the shape functions
        
        x, space = gf2array(w[self.Label])
        valGf = [GridFunction(space) for i in range(nChildren)]
        for i in range(nChildren):
            valGf[i].vec.FV().NumPy()[:] = self.Penalization[i].eval(x[:,i])
            
        # 3) Result
        result = 0
        for i in range(nChildren):
            result = valGf[i]*coeff[i] + result
        if flagCompil: return result.Compile()
        else : return result
    
    def evaldx(self, x, u, w_=dict(), dwdx_=dict(), k=1, result_=dict(), flagCompil = True):
        result = result_.copy()
        if len(w_) == 0 or len(dwdx_) == 0 : w, dwdx = self.evalBasisFunction(x)
        else: w, dwdx = w_.copy(), dwdx_.copy()
        
        if isinstance(x[self.Label], list): # NGsolve
            result = self.__evaldx_NGsolve(x, u, w, dwdx, k, result, flagCompil = True)
        else:
            if len(u.shape) < 3 : u= u.T.reshape(self.DimInput,1,-1)
            sz = u.shape[2]
            nChildren = len(self.Children)
            d1 = self.Children[0].DimOutput
            
            # 1) computation of the values of the children
            coeff = np.zeros((d1,1,sz,nChildren));
            for i in range(nChildren):
                child = self.Children[i]
                if isinstance(child, Interpolation):
                    coeff[:,:,:,i] = child.eval(x, u, w)
                    Pw = self.Penalization[i].eval(w[self.Label][i,:,:])
                    result = child.evaldx(x, u, w, dwdx, k*Pw, result)
                else:
                    coeff[:,:,:,i] = child.eval(u)
                
            # 2) computation of the derivative of the penalized shape functions
        
            dPdw = np.zeros((1,1,sz,nChildren))
            for i in range(nChildren):
                dPdw[0,0,:,i] = self.Penalization[i].evald(w[self.Label][i,0,:]);
    
            # 3) result
            result[self.Label] = k * np.sum(dPdw * dwdx[self.Label][None,:,:,:].transpose(0,2,3,1) * coeff, axis = 3)
        return result
    
    def __evaldx_NGsolve(self, x, u, w_=dict(), dwdx_=dict(), k=1, result_=dict(), flagCompil = True):
        result = result_.copy()
        if len(w_) == 0 or len(dwdx_) == 0 : w, dwdx = self.evalBasisFunction(x)
        else: w, dwdx = w_.copy(), dwdx_.copy()
        
        nChildren = len(self.Children)
        
        # 1) computation of the values of the children
        coeff = [None for i in range(nChildren)]
        x, space = gf2array(w[self.Label])
        for i in range(nChildren):
            child = self.Children[i]
            if isinstance(child, Interpolation):
                coeff[i] = child.eval(x, u, w)
                Pw =GridFunction(space)
                Pw.vec.FV().NumPy()[:] = self.Penalization[i].eval(x[:,i])
                result = child.__evaldx_NGsolve(x, u, w, dwdx, k*Pw, result)
            else:
                coeff[i] = child.eval(u)
            
        # 2) computation of the derivative of the penalized shape functions
    
        dPdw = [GridFunction(space) for i in range(nChildren)]
        for i in range(nChildren):
            dPdw[i].vec.FV().NumPy()[:] = self.Penalization[i].evald(x[:,i]);

        # 3) result
        dim = len(dwdx[self.Label][0])
        s = [0 for i in range(dim)]
        for i in range(nChildren):
            for j in range(dim):
                s[j] = dPdw[i] * dwdx[self.Label][i][j] * coeff[i] + s[j]
                
        result[self.Label] = [k * s[i] for i in range(dim)]
        #result[self.Label] = k * np.sum(coeff * dwdx[self.Label][None,:,:,:].transpose(0,2,3,1) * dPdw, axis = 3)
        if flagCompil : 
            for i in range(dim):
                result[self.Label][i] = result[self.Label][i].Compile()
        return result
    
    ###################
    #display
    ###################
    
    def plotTree(self, x= 0., ymin = -1., ymax = 1., scaleChildren = 1, cmap = plt.cm.Set1):
        if "3d" in str(type(plt.gca())): plt.figure()
        nChild = len(self.Children)
        interval = (ymax - ymin) / nChild
        yMean = (ymax + ymin)/2
        newX = x + 1
        plt.text(x, yMean+0.01, self.Label+"\n("+str(self.ShapeFunction.Domain.Dimension)+"D)")
        plt.plot(x, yMean, 'o', color = cmap(int(x) % 9))
        Xplot = np.vstack([[x* np.ones(nChild)],[newX* np.ones(nChild)]])
        Yplot = np.vstack([[yMean * np.ones(nChild)],[scaleChildren*(ymin + interval/2 + interval*np.arange(0, nChild))]])
        plt.plot(Xplot,Yplot, '--', color = cmap(int(x) % 9))
        for i in range(nChild):
            ym = ymin + interval*i
            yM = ymin + interval*(i+1)
            yMean = (ym+yM)/2
            if isinstance(self.Children[-1-i], Interpolation):
                self.Children[-1-i].plotTree(x = newX,  ymin = ym*scaleChildren, ymax = yM*scaleChildren, scaleChildren=scaleChildren)
            else:
                plt.plot(newX, yMean*scaleChildren, 'o', color = cmap(int(newX) % 9))
                plt.text(newX+0.01, yMean*scaleChildren, str(self.Children[-1-i]))
        
        plt.axis("off")
           
    
    def plot(self, x = dict(), level=0, offset = np.array([[0,0,0]]) ,
             distance = lambda x: 0, direction = np.array([[0,0,0]]),
             cmap = plt.cm.Set1, linewidth = 1, indexPlot = "all"):
        if "3d" not in str(type(plt.gca())): plt.close() ; plt.figure();  plt.gcf().add_subplot(projection='3d')
        
        if len(x) == 0: xyz = np.zeros((0,3))
        elif isinstance(x[self.Label] , list) : 
            xyz = [gf.vec.FV().NumPy()[:].reshape(-1,1) for gf in x[self.Label]]
            xyz = np.hstack([*xyz, np.zeros((xyz[0].shape[0],3))])[:,0:3]
        else: xyz = np.hstack([x[self.Label], np.zeros((x[self.Label].shape[0],3))])[:,0:3]
        
        xyz3d = np.ones(xyz.shape)*offset;
        xyz3d[:,0:xyz.shape[1]] =  xyz3d[:,0:xyz.shape[1]] + xyz;
        if isinstance(indexPlot, (slice,int)) and xyz3d.size>0 : xyz3d = xyz3d[indexPlot,:].reshape(-1,3)
        c = cmap(level)
        d = self.ShapeFunction.Domain
        v = d.Vertices
        
        v3d = np.ones((v.shape[0],3))*offset
        v3d[:,0:v.shape[1]] =  v3d[:,0:v.shape[1]] + v
        fList = []
        
        for i in range(1,v.shape[1]):
            fList.append(lambda p: (distance(p[i,:])-distance(p[i-1,:]))**2)
        
        fmin = lambda p: np.sum([f(p) for f in fList]) + (distance(p[0,:])-distance(p[-1,:]))**2

        fobj = lambda th : fmin(rotPoint((v3d-offset).T,th[0],th[1],th[2]).T+offset)
        thOpt = minimize(fobj,np.ones(3)*1e-2)
        thOpt = thOpt.x
        v3d = rotPoint((v3d-offset).T,thOpt[0],thOpt[1],thOpt[2]).T+offset
        xyz3d = rotPoint((xyz3d-offset).T,thOpt[0],thOpt[1],thOpt[2]).T+offset

        dim = d.Dimension
        xyzText = (offset+direction*0.15).squeeze()
        plt.gca().text(xyzText[0],xyzText[1],xyzText[2],self.Label)  
        dCopy = deepcopy(d)
        dCopy.Vertices = v3d
        if dim == 1: dCopy.plot(color = c, linewidth = linewidth*2, textFlag = False)
        else: dCopy.plot(color = c, linewidth = linewidth, textFlag = False)
        
        plt.gca().scatter(xyz3d[:,0],xyz3d[:,1],xyz3d[:,2],marker = 'x', color = 'k')
        
        bc = np.mean(v3d,axis = 0);
        for i in range(len(self.Children)):
            child = self.Children[i]
            if isinstance(child, Interpolation):
                offset = 3*(v3d[i,:]-bc)+bc
                direction = (v3d[i,:]-bc)/np.linalg.norm(v3d[i,:]-bc)
                
                plt.gca().plot(np.vstack([v3d[i,0],offset[0]]),
                               np.vstack([v3d[i,1],offset[1]]),
                               np.vstack([v3d[i,2],offset[2]]),'k--')
                
                child.plot(x,level+1,offset,lambda p : np.linalg.norm(p-bc), direction,
                           cmap = cmap, linewidth = linewidth, indexPlot = indexPlot)
            else:
                a = np.linalg.norm(v3d[i,:]-bc)
                if a!=0:
                    dir2 = 0.08*(v3d[i,:]-bc)/a
                else:
                    dir2 = 0.08*direction
                
                l = child.Label + "\n"
                xyzText = v3d[i] + dir2
                plt.gca().text(xyzText[0],xyzText[1],xyzText[2], l, color = c);
            plt.gca().view_init(elev=30, azim=45)
            plt.axis("equal")
            plt.axis("off")
    
    
    def getAllNodes(self, nodes = None):
        if nodes is None : n = []
        
        n.append(self)
        for child in self.Children:
            if isinstance(child, Interpolation):
                n = child.getAllNodes()
        return n
    
    def __str__(self, space = 0):
        
        txt = (space-1)*"|   " + "|--" + f'"{self.Label}" [ {self.ShapeFunction.Domain.Dimension}D {self.ShapeFunction.Domain.Type} ] \n'
        txt += (space+1)*"|   " + "\n"
        for i in range(len(self.Children)):
            child = self.Children[i]
            if isinstance(child, Interpolation):
                txt += space*"|   " + f"|  ({i+1}) Penalization : {self.Penalization[i].Type}, p = {self.Penalization[i].CoeffPenal}\n"
                #txt += (space+2)*"|   " + "\n"
                txt += child.__str__(space+1)
            else :
                txt += space*"|   " + f"|   ({i+1}) ( Penalization : {self.Penalization[i].Type}, p = {self.Penalization[i].CoeffPenal} ) * "
                txt += f"( {child.Label} : u in R^{child.DimInput} -> R^{child.DimOutput} ) \n"
        txt += (space+1)*"|   " + "\n"
        return txt

        
    
'''
II) Shape functions
'''

class ShapeFunction:
    
    def __init__(self, domain, label = "", reference = "", type= "wachspress"):
        self.Type = type
        self.Domain = domain
        self.Reference = None
        self.Label = label
        
        if type.lower()=="w" or type.lower()=="wachspress":
            self.Type = "wachspress"
            self.Reference = "Wachspress, 1975 ; Floater et. al. 2014";
        else:
            raise Exception(f"Basis function type '{type}' not supported")
            
            
    
    def eval(self, x):
        """
        Evaluate the shape functions w and their gradient dwdx for a given position x inside the domain.
        
        Parameters:
        - x : numpy array of shape (n, dim)
        
        Returns:
        - w : numpy array of shape (n, dim)
        - dwdx : numpy array of shape (n, dim, dim)
        """
        if self.Type == "wachspress":
            
            dim = self.Domain.Dimension
            if dim>1 : assert x.shape[1] == dim, "Wrong input dimension"
            
            if dim == 0:
                w = np.ones((1, 1, x.shape[0]))
                dwdx = np.zeros((1, 0,  x.shape[0]))
            elif dim == 1:
                x = np.expand_dims(x, axis=(0, 1))
                w = w = np.concatenate([-x, x], axis=1).squeeze().reshape(2,1,-1) + 0.5  # assuming centering on 0
                dwdx = np.concatenate([-np.ones((1,1,x.size)),np.ones((1,1,x.size))], axis=0)
            elif dim == 2:
                w, dwdx = wachspress2d(x, self.Domain)
            elif dim == 3:
                w, dwdx = wachspress3d(x, self.Domain)
            else:
                raise ValueError("incompatible density field")
                # If you want to add other types of basis functions,
                # follow the same model as the Wachspress' one.
                # Please vectorize the code to maximize performance (avoid for loop)
        else:
            raise ValueError(f"Basis function type {self.Type} not supported")

        return w, dwdx
    
    def plot(self, nVertex = 0, level=3,  cmap = plt.cm.viridis):
        eps = self.Domain.NormalFan.EpsProj
        vCentered=(self.Domain.Vertices-np.mean(self.Domain.Vertices,axis=0))*(1-eps)
        
        if self.Domain.Dimension == 0: # 0D
            plt.scatter(0,1)
            plt.grid()
            plt.xlabel('x')
            plt.ylabel(r'$\omega(x)$')
            plt.text(0 , 1, "v0")
            plt.show()
            
        if self.Domain.Dimension == 1: # 1D
            x = np.linspace(vCentered[0][0], vCentered[1][0], 100)
            w, _ = self.eval(x)
            w = w[nVertex]
            plt.plot(x, w.flatten())
            plt.xlabel('x')
            plt.ylabel(r'$\omega(x)$')
            for i in range(vCentered.shape[0]):
                plt.text(vCentered[i] , -0.05, f"v{i}")
            plt.plot(vCentered,[0,0],'k--|')
            plt.grid()
            plt.axis((vCentered[0] * 1.1, vCentered[1] * 1.1, -0.1, 1))
            plt.show()
            
        elif  self.Domain.Dimension == 2: # 1D
            npp = vCentered.shape[0]
            p = [[np.array([0,0]), vCentered[i], vCentered[ (i+1) % npp ]] for i in range(npp)]
            pointsTri = np.zeros((0,2))
            for pp in p:
                pointsTri = np.append(pointsTri, np.unique(pointTriangle(pp, l=level, p=[]), axis=0),axis=0)
            pointsTri = np.unique(pointsTri, axis=0)
            
            x = pointsTri[:,0]
            y = pointsTri[:,1]
            z, _ =  self.eval(pointsTri)
            z = z[nVertex,:,:].flatten()
            
            
            ax = plt.figure().add_subplot(projection='3d')
            ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, cmap=cmap )
            ax.plot(np.append(vCentered[:,0],[vCentered[0,0]]),
                    np.append(vCentered[:,1],[vCentered[0,1]]),
                    np.zeros(vCentered.shape[0]+1),'k--|')
            for i in range(npp):
                ax.text(vCentered[i,0]*1.1, vCentered[i][1]*1.1, 0.08, f"v{i}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("$\omega(x,y)$")
            plt.show()
            
        elif self.Domain.Dimension == 3: # 3D (only the facets are plotted)
            ax = plt.figure().add_subplot(projection='3d')
            for facets in self.Domain.Facets:
                pointsFacets = []
                for edges in facets:
                    if edges>0 : pointsFacets.append(self.Domain.Edges[edges-1][0])
                    else : pointsFacets.append(self.Domain.Edges[abs(edges)-1][1])
                
                # mapping facet -> reference to perform the 2D mesh
                
                xyzFacet = np.array(vCentered[pointsFacets,:])   # coordinates of the facet's points              
                vc = np.mean(xyzFacet, axis = 0)                 # center of the facet 
                
                # decomposition in triangles
                npp = len(pointsFacets)
                pTriangle = [[vc, xyzFacet[i], xyzFacet[ (i+1) % npp]] for i in range(npp)]
                pointsTri = np.zeros((0,3))
                
                for pp in pTriangle: # "mesh" every elementary triangles of the facet
                    
                    # 1) map to the reference elt
                    J = np.array([pp[0],pp[1],pp[2]]).reshape(3,3).T
                    invJ = np.linalg.inv(J)
                    Ref2Facet = lambda x: mult(J,x.T)
                    Facet2Ref = lambda x: mult(invJ,x.T)
                    
                    ptRef = Facet2Ref(np.array(pp))
                
                    pointsTriRefLocal = np.unique(pointTriangle(ptRef, l=level), axis=0)
                    
                    tri = Triangulation(pointsTriRefLocal[:,0],pointsTriRefLocal[:,1])
                    
                    pointsTriLocal = Ref2Facet(pointsTriRefLocal)
                    w, _ =  self.eval(pointsTriLocal.T)
                    w = w[nVertex,:,:].flatten()
                    p3dc = ax.plot_trisurf(pointsTriLocal[0], pointsTriLocal[1], pointsTriLocal[2],
                                    triangles=tri.triangles, linewidth=0.5,  color = [1,1,1,0.9])
                    #p3dc.set_fc(plt.get_cmap(cmap)(w))    
                ax.set_aspect('equal', 'box')
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
        
        
    def print(self):
        print(f'{self.Type} basis functions defined on a {self.Domain.Dimension}D'
              + f' domain with {self.Domain.Vertices.shape[0]} vertices')
    
    
def getNormals(v):
    """
    Compute the outward unit normals for each edge of a polygon.
    
    This function calculates the unit normal vectors for each edge of a polygon
    given its vertices. The normal vectors are oriented outward from the polygon.
    
    Args:
    vertices (numpy.ndarray): An (n, 2) array of shape (n, 2), where n is the
                              number of vertices. Each row represents the
                              coordinates of a vertex (x, y) of the polygon.
    
    Returns:
    numpy.ndarray: An (n, 2) array of unit normal vectors, where each row
                   corresponds to the outward normal of an edge. The normal
                   vector is computed for each edge defined by consecutive
                   vertices, with the last vertex connecting back to the first.
    
    Example:
    >>> import numpy as np
    >>> vertices = np.array([
    ...     [0.0, 0.0],  # Vertex 1
    ...     [1.0, 0.0],  # Vertex 2
    ...     [1.0, 1.0],  # Vertex 3
    ...     [0.0, 1.0]   # Vertex 4
    ... ])
    >>> normals = get_normals(vertices)
    >>> print(normals)
    [[ 0. -1.]
     [ 1.  0.]
     [ 0.  1.]
     [-1.  0.]]
    
    Notes:
    - The function assumes the vertices are ordered in a way that defines the
      edges of the polygon.
    - The normal vectors are computed using the cross product of vectors defining
      each edge, normalized to unit length.
    
    """
    n = v.shape[0]
    un = np.zeros((n, 2))
    ind1 = np.mod(np.arange(1, n + 1), n)
    ind2 = np.arange(n)
    d = v[ind1] - v[ind2]
    un[ind2] = np.stack([d[:, 1], -d[:, 0]], axis=1) / np.linalg.norm(d, axis=1, keepdims=True)
    return un


def wachspress2d(rho, domain2d):
    """
    Evaluate Wachspress basis functions and their gradients in a convex polygon.

    Args:
        rho (numpy.ndarray): An (np, 2) array where each row represents a coordinate
                             where the shape functions should be computed.
        domain2d (object): An object with attributes 'Vertices' representing the 2D domain.

    Returns:
        w (numpy.ndarray): Basis functions as an array of shape (np, n).
        dwdrho (numpy.ndarray): Gradient of basis functions as an array of shape (np, n, 2).
    
    Note:
    based on the Matlab code from :
        M. Floater, A. Gillette, and N. Sukumar,
        “Gradient bounds for Wachspress coordinates on polytopes,”
        SIAM J. Numer. Anal., vol. 52, no. 1, pp. 515–532, 2014,
        doi: 10.1137/130925712.
    """
    v = domain2d.Vertices
    if len(np.array(rho).shape) == 0 : rho = np.array(rho)[None]
    else : rho = np.array(rho)
    np_points = rho.shape[0]
    nv = v.shape[0]

    un = np.reshape(getNormals(v).T, (2, 1, nv,1))
    h = mult(t(v).reshape(1,2,nv,1) - t(rho).reshape(1,2,1,np_points),un)
    p = un / h

    i = np.arange(nv)
    im1 = (i - 1) % nv
    w = p[0, :, im1, :] * p[1, :, i, :] - p[0, :, i, :] * p[1, :, im1, :]
    wsum = np.sum(w, axis=0, keepdims=True)

    R = np.transpose(p[:, :, im1, :] + p[:, :, i, :], (2, 0, 3, 1))
    w = (w / wsum).reshape(nv,1,np_points,1)
    phiR = mult(t(w), R)
    dwdrho = (w * (R - phiR))

    return w.reshape(nv,1,np_points), dwdrho.squeeze().reshape(nv,2,np_points)


def wachspress3d(rho, domain3d):
    """
    Evaluate Wachspress basis functions and their gradients in a convex polyhedron.

    Args:
        rho (numpy.ndarray): An (np, 3) array where each row represents a coordinate
                             where the shape functions should be computed.
        domain3d (object): An object with attributes 'Vertices', 'Normals', and 'Vertices2Facets'
                            representing the 3D domain.

    Returns:
        w (numpy.ndarray): Basis functions as an array of shape (n, np).
        dwdrho (numpy.ndarray): Gradient of basis functions as an array of shape (n, np, 3).
    """
    if len(np.array(rho).shape) == 0 : rho = np.array(rho)[None]
    else : rho = np.array(rho)
    np_points = rho.shape[0]
    v = domain3d.Vertices
    un = domain3d.Normals
    g = domain3d.Vertices2Facets

    n = v.shape[0]
    w = np.zeros((n,1 , np_points))
    R = np.zeros((n, 3, np_points))

    rho = np.reshape(rho.T, (1, 3, 1, np_points))
    un = np.reshape(un.T, (1, 3, -1))
    v = np.reshape(v.T, (1, 3, -1))

    for i in range(n):
        f = g[i]
        k = len(f)-1
        h = mult(v[:,:,i,None,None] - rho, t(un[:,:,f,None]))
        p = np.transpose(un[:, :, f, None] / h, (0, 1, 3, 2))
        
        j = np.arange(k - 1)
        wloc = p[0, 0, :, j] * (p[0, 1, :, j + 1] * p[0, 2, :, k] - p[0, 1, :, k] * p[0, 2, :, j + 1]) + p[0, 0, :, j + 1] * (p[0, 2, :, j] * p[0, 1, :, k] - p[0, 2, :, k] * p[0, 1, :, j]) +p[0, 0, :, k] * (p[0, 1, :, j] * p[0, 2, :, j + 1] - p[0, 1, :, j + 1] * p[0, 2, :, j])
        wloc = wloc[:,None,:]
        Rloc = (p[:, :, :, j] + p[:, :, :, j + 1] + p[:, :, :, [k]]).transpose(3,1,2,0)[:, :, :, 0]
        w[i, :] = np.sum(wloc, axis=0, keepdims=True)
        R[i, :, :] = (mult(t(wloc), Rloc) / w[i,:,:])

    wsum = np.sum(w, axis=0, keepdims=True)
    w = w / wsum
    phiR = mult(t(w),R)
    dwdrho = w * (R - phiR)

    return w, dwdrho
