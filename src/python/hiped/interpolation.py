# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 18:51:38 2024

@author: tcherriere
"""

from .utils import t, mult
from .vertexFunction import VertexFunction
from .domain import *
from .penalization import *
from .vizualization import *

import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.tri import Triangulation
import matplotlib.colors as mcolors
from scipy.optimize import minimize
from copy import deepcopy

# %% I) Interpolation

class Interpolation:
    """
    A class used to represent an Interpolation
    
    An Interpolation object describes a mathematical tree. The leaves are
    VertexFunctions, interpolated by ShapeFunctions and Penalizations. See
    the examples files for some illustrations.


    Attributes
    ----------
        
    Label : str
        Label of the current interpolation node (should be unique in the 
        interpolation tree)
    
    Children : list
        The list of children (VertexFunction or Interpolation instances)

    ShapeFunction : ShapeFunction
        Description of the shape functions associated to the current node 
        (default : Wachspress basis functions)
    
    Penalization : list
        Penalization associated to each child of the interpolation (default : 
        Penalization("simp",1) for each child)
                    
    DimInput : int
        Dimension of the input "physical field" u (default 1 => scalar)
    
    DimOutput : int
        Dimension of the resulting interpolated quantity (default 1 => scalar)


    Methods
    -------
    
    setInitialVariable(nVariables = 10, typeInit = "zero", radius = 1, 
                       x00 = dict(), NGSpace = None)
        Set the initial coordinates with the appropriate structure
    
    projection(x, copyFlag = False, deepcopyFlag = False)
        Project the coordinates contained in x onto the interpolation domain
        
    evalBasisFunction(x)
        Compute the basis functions associated to each node of the 
        interpolation tree from the coordinates x
        
    eval(x, u, w=dict())
        Compute the interpolation associated to the coordinates x and field u
    
    evaldu(x, u, w=dict())
        Compute the derivative of the interpolation with respect to the field u
    
    evaldx(x, u, w=dict(), dwdx=dict())
        Compute the derivative of the interpolation with respect to the
        coordinates x
        
    plotTree(x= 0, ymin = -1, ymax = 1, scaleChildren = 1, cmap = plt.cm.Set1)
        Plot a tree representation of the interpolation
    
    plot(x = dict())
        Plot a 3D representation of the interpolation, possibly with the
        coordinate x if provided
        
    getAllNodes()
        Return all the Interpolation instances of the progeny (including 
        current node)
    
    License
    -------
    
    Copyright (C) 2024 Théodore CHERRIÈRE (theodore.cherriere@ricam.oeaw.ac.at)
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
    """

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
    
    def setInitialVariable(self, nVariables, typeInit = "zero", radius = 1, x00 = dict()):
        if x00 == dict(): x0 = x00.copy() 
        else: x0 = x00
        assert self.Label not in x0.keys(), "Non-unique interpolation labels"
        dim = self.ShapeFunction.Domain.Dimension
        
        if typeInit.lower() in ["zero", "zeros", "z", "null", "center"]:
            x0[self.Label] = np.zeros((nVariables, dim))
            
        elif typeInit.lower() in ["rand", "random", "r"]:
            if dim == 0:
                x0[self.Label] = np.zeros((nVariables, dim))
            elif dim == 1:
                x0[self.Label] = radius*(np.random.rand(nVariables, dim)-0.5)
            elif dim == 2:
                R = radius*np.random.rand(nVariables)
                th = np.random.rand(nVariables)*2*np.pi
                x, y = R * np.cos(th), R * np.sin(th)
                x0[self.Label] = np.array([x,y]).T
            elif dim == 3:
                R = radius*np.random.rand(nVariables)
                th = np.random.rand(nVariables)*2*np.pi
                phi = np.random.rand(nVariables)*2*np.pi
                x = R * np.cos(th)* np.cos(phi)
                y = R * np.cos(th)* np.sin(phi)
                z = R * np.sin(th)
                x0[self.Label] = np.array([x,y,z]).T
        
        for child in self.Children:
            if isinstance(child,Interpolation):
                x0 = child.setInitialVariable(nVariables, typeInit, radius, x0)
                
        return x0
    
    
    def projection(self, x, xproj = None):
        if xproj is None : xproj = x
        xproj[self.Label] = self.ShapeFunction.Domain.projection(x[self.Label])
        for child in self.Children:
            if isinstance(child, Interpolation):
                xproj = child.projection(x, xproj)
        return xproj
    
    ''' 
    Evaluation
    '''
    
    def evalBasisFunction(self, x, w=dict(), dwdx=dict()):
        w[self.Label], dwdx[self.Label] = self.ShapeFunction.eval(x[self.Label])
        for child in self.Children:
            if isinstance(child, Interpolation):
                w, dwdx = child.evalBasisFunction(x, w, dwdx)
                
        return w, dwdx
    
    
    def eval(self, x, u, w_=dict()):
        # x = dict
        # u = dimInput x 1 x N
        if len(w_) == 0: w, _ = self.evalBasisFunction(x)
        else: w = w_.copy()
        
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
        return np.sum(coeff * val, 3)
        

    def evaldu(self, x, u, w_=dict()):
        if len(w_) == 0: w, _ = self.evalBasisFunction(x)
        else: w = w_.copy()
        
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
        return np.sum(coeffd*Pw, 3)
    
    def evaldx(self, x, u, w_=dict(), dwdx_=dict(), k=1, result_=dict()):
        result = result_.copy()
        if len(w_) == 0 or len(dwdx_) == 0 : w, dwdx = self.evalBasisFunction(x)
        else: w, dwdx = w_.copy(), dwdx_.copy()
        
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
        result[self.Label] = k * np.sum(coeff * dwdx[self.Label][None,:,:,:].transpose(0,2,3,1) * dPdw, axis = 3)
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
           
    
    def plot(self,x = dict(), level=0, offset = np.array([[0,0,0]]) ,
             distance = lambda x: 0, direction = np.array([[0,0,0]]),
             cmap = plt.cm.Set1, linewidth = 1, indexPlot = "all"):
        if "3d" not in str(type(plt.gca())): plt.close() ; plt.figure();  plt.gcf().add_subplot(projection='3d')
        
        if len(x) == 0: xyz = np.zeros((0,3))
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

        
    
# %% II) Shape functions

class ShapeFunction:
    """
    A class used to represent an ShapeFunction
    
    A ShapeFunction object is the elementary brick to build interpolations on
    general domain (restricted to convex polyedrons up to 3 dimensions). It
    maps the cartesian coordinate x to one scalar value in [0,1] per domain 
    vertex. Since they have the appropriate mathematical properties
    (generalized barycentric coordinate, see [1], [2]), the resulting shape 
    functions (denoted as w) can then be multiplied by the quantity to
    interpolate and before being summed up.
    
    [1] Floater, M. S. (2015). Generalized barycentric coordinates and 
    applications. In Acta Numerica (Vol. 24, pp. 161–214). Cambridge University 
    Press (CUP). https://doi.org/10.1017/s0962492914000129 
    
    [2] Gillette, A. (2011) Generalized Barycentric Coordinates for Polygonal
    Finite Elements https://math.arizona.edu/~agillette/research/ccomOct11.pdf

    Attributes
    ----------
    
    Domain : Domain
        Interpolation domain (instance of Domain class)
    
    Type : str
        Type of shape function. For now only "wachspress" is supported; other
        types exist but are not implemented yet (such as mean-value, Malsch,
        etc.), see for instance:
            
        Floater, M. S., Hormann, K., & Kós, G. (2006). A general construction 
        of barycentric coordinates over convex polygons. In Advances in 
        Computational Mathematics (Vol. 24, Issues 1–4, pp. 311–331). Springer 
        Science & Business Media LLC. https://doi.org/10.1007/s10444-004-7611-6 
        
    Reference : str
        Literature reference(s) related to the shape function origine or its
        implementation
    
    Methods
    -------
    
    eval(x)
        Evaluate the shape functions and their gradient at the cartesian
        coordinate(s) x, that should lie inside the interpolation domain
    
    plot(nVertex = 0, level=3,  cmap = plt.cm.viridis)
        Plot a representation of the shape function associated to the vertex
        number nVertex.
    
    
    License
    -------
    
    Copyright (C) 2024 Théodore CHERRIÈRE (theodore.cherriere@ricam.oeaw.ac.at)
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
    """

    
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
            X, Y, Z = np.zeros(0),np.zeros(0),np.zeros(0)
            FC = np.zeros(0)
            TRI = np.zeros((0,3), dtype = int)
            sz = 0
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
                
                    # 2) mesh it efficiently
                    
                    xx, yy, tri = meshTri(2**level+1)
                    TRI = np.vstack([TRI, tri+sz])
                    xx = xx.reshape(1,-1)
                    yy = yy.reshape(1,-1)
                    zz = xx*ptRef[-1,0] + yy*ptRef[-1,1] + (1-xx-yy)*ptRef[-1,2]
                    pointsTriRefLocal = np.vstack([xx,yy,zz]).T
                        
                    # 3) Pull back to the original space
                    pointsTriLocal = Ref2Facet(pointsTriRefLocal)
                    X = np.hstack([X,pointsTriLocal[0,:]])
                    Y = np.hstack([Y,pointsTriLocal[1,:]])
                    Z = np.hstack([Z,pointsTriLocal[2,:]])
                    sz = len(X)
                    
                    # 4) compute shape functions
                    w, _ =  self.eval(pointsTriLocal.T)
                    w = w[nVertex,:,:].flatten()
                    
                    FC =  np.hstack([FC,np.sum(w[tri], axis = 1)/3]) 
                    
            p3dc = ax.plot_trisurf(X, Y, Z, triangles=TRI, linewidth=0.5,  color = [1,1,1,0.9])
            p3dc.set_fc(plt.get_cmap(cmap)(FC))    
            ax.set_aspect('equal', 'box')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        
        
    def print(self):
        print(f'{self.Type} basis functions defined on a {self.Domain.Dimension}D'
              + f' domain with {self.Domain.Vertices.shape[0]} vertices')
    
    
# %% III) Utilities


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

#%%

# dimInput = 2
# dimOutput = 2

# f1v = VertexFunction(label = "f1v", f = lambda u : u**2, dfdu = lambda u : 2*u * np.eye(2)[:,:,None], dimInput= dimInput, dimOutput = dimOutput)
# f2v = VertexFunction("f2v", lambda u : mult(np.array([[1,2],[3,4]]), u), lambda u : np.array([[1,2],[3,4]])[:,:,None] * np.ones(u.shape), dimInput, dimOutput)
# f3v = VertexFunction("f3v", lambda u : 0.1*u, lambda u : 0.1 * np.eye(2)[:,:,None] * np.ones(u.shape), dimInput, dimOutput)

# fMat =  VertexFunction("f3v", lambda u : np.array([[1,2],[3,4]]).reshape(2,2,1)*np.ones(u.shape), lambda u : np.zeros(u.shape) * np.zeros((2,2,1)), dimInput, (2,2))
# fConst =  VertexFunction("f3v", lambda u : np.ones((1,1,1))*np.ones(u.shape), lambda u : np.zeros((1,1,1))*np.zeros(u.shape), dimInput, 1)
# fConst2 =  VertexFunction("f3v", lambda u : np.ones((2,1,1))*np.ones(u.shape), lambda u : np.zeros((2,2,1))*np.zeros(u.shape), dimInput, 2)

# f4v = 3* f1v.innerProduct(f2v)*f3v
# f5v = fMat @ f1v 


# f1v = VertexFunction(label = "f1v", f = lambda u : u**2, dfdu = lambda u : 2*u * np.eye(2)[:,:,None], dimInput= dimInput, dimOutput = dimOutput)
# f2v = VertexFunction("f2v", lambda u : mult(np.array([[1,2],[3,4]]), u), lambda u : np.array([[1,2],[3,4]])[:,:,None] * np.ones(u.shape), dimInput, dimOutput)
# f3v = VertexFunction("f3v", lambda u : 0.1*u, lambda u : 0.1 * np.eye(2)[:,:,None] * np.ones(u.shape), dimInput, dimOutput)

# fMat =  VertexFunction("fMat", lambda u : np.array([[1,2],[3,4]]).reshape(2,2,1)*np.ones(u.shape), lambda u : np.zeros(u.shape) * np.zeros((2,2,1)), dimInput, (2,2))
# fConst =  VertexFunction("fConst", lambda u : np.ones((1,1,1))*np.ones(u.shape), lambda u : np.zeros((1,1,1))*np.zeros(u.shape), dimInput, 1)
# fConst2 =  VertexFunction("fConst2", lambda u : np.ones((2,1,1))*np.ones(u.shape), lambda u : np.zeros((2,2,1))*np.zeros(u.shape), dimInput, 2)

# f4v = fMat @ f1v * (f3v+f2v) / 2 # only multiplication by constant matrix is supported
# print(f4v)
# f5v = 3* f1v.innerProduct(f2v)*f3v 
# print(f5v)

# interpVector= Interpolation(domain = Domain(5), children = [f1v,f2v,f3v,f4v,f5v],
#                                label = "interpScalar", penalization=Penalization("simp", 2))

# x = interpVector.setInitialVariable(100,"rand",1)
# x = interpVector.projection(x)
# u = np.random.rand(2,1,100)
# w, dwdx = interpVector.evalBasisFunction(x)
# f = interpVector.eval(x, u, w) # value
# dfdu =  interpVector.evaldu(x, u, w) # derivative w.r.t a
# dfdx =  interpVector.evaldx(x, u, w, dwdx) # derivative w.r.t x


# # derivative with respect to u
# h = np.logspace(-8,-2,10) # test for 10 different h
# resU = np.zeros((10,1,u.shape[2]))


# for i in range(10):
#     pert = h[i]*np.random.rand(*u.shape)
#     fPerturbedu = interpVector.eval(x,u+pert)
#     resU[i,0,:] = np.linalg.norm(fPerturbedu - (f + mult(dfdu,pert)), axis = 0)
    
# maxResU = np.max(resU, axis = 2); medResU = np.median(resU, axis = 2);

# plt.figure()
# plt.loglog(h, medResU,'-o', label =  "median of the residual")
# plt.loglog(h, maxResU,'-o', label =  "maximum of the residual")
# plt.loglog(h, h**2,'k--', label =  "expected decay $(h^2)$")

# plt.legend(loc = "best"); plt.grid()
# plt.xlabel("$h$"); plt.ylabel("Euclidian norm of Taylor remainder")
# plt.title("Taylor remainder with respect to $u$")

# plt.show()