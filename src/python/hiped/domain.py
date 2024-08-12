# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:40:40 2024

@author: tcherriere
"""

from .utils import mult, t

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Polygon
from  matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class Domain:
    def __init__(self, domain_type, radius=1, lengthNormalFan = 1, epsProj=1e-5):
        
        self.Vertices = None
        self.Edges = None
        self.Facets = None
        self.Normals = None
        self.Vertices2Facets = None
        self.Edges2Facets = None
        self.Dimension = None
        self.NormalFan = None
        
        
        if isinstance(domain_type, dict): # Custom domain
            self.Vertices = np.array(domain_type['vertices'])
            self.Edges = np.array(domain_type['edges'])
            self.Facets = domain_type['facets']
            self.Normals = np.array(domain_type['normals'])
            self.Vertices2Facets = domain_type['vertices2facets']
            self.Edges2Facets = domain_type['edges2facets']
            self.Dimension = domain_type['dimension']
            self.NormalFan = domain_type['normal_fan']

        else: # Predefined domains
        
        # 0D domain (just a point)
            if isinstance(domain_type, (int, float)) and domain_type == 1:
                self.Vertices = np.array([[0]])
                self.Dimension = 0
                self.Type = "Dot"
                
        # 1D domain (a line)    
            elif isinstance(domain_type, (int, float)) and domain_type == 2:
                self.Vertices = np.array([[-0.5], [0.5]])
                self.Dimension = 1
                self.Type = "Line"
                
        # 2D domain (regular polygon)
            elif isinstance(domain_type, (int, float)) and domain_type > 2:
                theta = np.linspace(0, 2 * np.pi, int(domain_type) + 1) + (2 * np.pi) / (2 * domain_type)
                theta = theta[:-1]
                R = mult(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([[1], [0]]))
                self.Vertices = R.reshape(2, domain_type).T
                self.Dimension = 2
                self.Type = "RegularPolygon"
                
        # 3D domain (tetraedron, cube, diamond and prism)
            elif isinstance(domain_type, str) and domain_type.lower() in ["tetra", "tetraedron"]:
                self.Type = "Tetraedron"
                self.Dimension  = 3
                theta = np.linspace(0, 2 * np.pi, 3 + 1) + (2 * np.pi) / (2 * 3)
                theta = theta[:-1]
                h = 1/3 * radius
                R = mult(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([[radius * 2 * np.sqrt(2) / 3], [0]]))
                v1 = R.reshape(2, 3).T
                self.Vertices = np.concatenate([np.concatenate([v1, -np.ones((3, 1)) * h], axis = 1),
                                               [[0, 0, radius]]])
                self.Vertices -= np.mean(self.Vertices, axis=0)
                
                v = self.Vertices
                un = np.zeros((4, 3))
                un[0] = np.cross(v[3] - v[0], v[1] - v[0])
                un[1] = np.cross(v[3] - v[1], v[2] - v[1])
                un[2] = np.cross(v[3] - v[2], v[0] - v[2])
                un[3] = np.cross(v[1] - v[0], v[2] - v[0])
                self.Normals = -un / np.linalg.norm(un, axis=1)[:, None]

                self.Vertices2Facets = [[0, 2, 3], [1, 0, 3], [2, 1, 3], [0, 1, 2]] # indexed from 0
                self.Edges = [[0, 1], [1, 2], [2, 0], [3, 0], [3, 1], [3, 2]]       # indexed from 0
                self.Facets = [[1, -5, 4], [2, -6, 5], [3, -4, 6], [-1, -3, -2]]
                self.Edges2Facets = np.zeros((len(self.Edges), 2), dtype=int)
                
                for i, facet in enumerate(self.Facets):
                    edges = np.abs(facet)
                    for edge in edges:
                        if self.Edges2Facets[edge - 1, 0] == 0:
                            self.Edges2Facets[edge - 1, 0] = i + 1
                        else:
                            self.Edges2Facets[edge - 1, 1] = i + 1
                self.Edges2Facets -= 1
                
            elif isinstance(domain_type, str) and domain_type.lower() == "cube":
                self.Type = "Cube"
                self.Dimension  = 3
                v = 2 * radius * np.sqrt(3) / 3 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], 
                                                            [1, 1, 0], [0, 0, 1], [1, 0, 1], 
                                                            [0, 1, 1], [1, 1, 1]])
                self.Vertices = v - np.mean(v, axis=0)
                self.Vertices2Facets = [[0, 3, 5], [1, 0, 5], [3, 2, 5], 
                                        [2, 1, 5], [4, 3, 0], [4, 0, 1], 
                                        [4, 2, 3], [4, 1, 2]] # indexed from 0


                un = np.zeros((6, 3))
                un[0] = np.cross(v[1], v[4])
                un[1] = np.cross(v[3] - v[1], v[5] - v[1])
                un[2] = np.cross(v[2] - v[3], v[7] - v[3])
                un[3] = np.cross(v[0] - v[2], v[6] - v[2])
                un[4] = np.cross(v[5] - v[4], v[6] - v[4])
                un[5] = np.cross(v[0] - v[1], v[3] - v[1])
                self.Normals = un / np.linalg.norm(un, axis=1)[:, None]

                self.Edges = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], 
                              [7, 6], [6, 4], [0, 4], [1, 5], [3, 7], [2, 6]] # indexed from 0
                
                
                self.Facets = [[1, 10, -5, -9], [2, 11, -6, -10], [3, 12, -7, -11],
                               [4, 9, -8, -12], [5, 6, 7, 8], [1, 2, 3, 4]]
                self.Edges2Facets = np.zeros((len(self.Edges), 2), dtype=int)
                
                for i, facet in enumerate(self.Facets):
                    edges = np.abs(facet)
                    for edge in edges:
                        if self.Edges2Facets[edge - 1, 0] == 0:
                            self.Edges2Facets[edge - 1, 0] = i + 1
                        else:
                            self.Edges2Facets[edge - 1, 1] = i + 1
                self.Edges2Facets -= 1

            elif isinstance(domain_type, str) and "diamond" in domain_type.lower():
                self.Dimension  = 3
                n = int(''.join(filter(str.isdigit, domain_type)))
                self.Type = "Diamond"+str(n)
                theta = np.linspace(0, 2 * np.pi, n + 1) + np.pi / n
                theta = theta[:-1]
                
                R = mult(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([[radius], [0]]))
                v = R.reshape(2, n).T
                
                v = np.vstack([np.hstack([v, np.zeros((n, 1))]), [0, 0, radius], [0, 0, -radius]])
               
                self.Vertices = v - np.mean(v, axis=0)
                v = self.Vertices
                
                self.Vertices2Facets = []
                for i in range(1, n+1):
                    i1 = i - 1
                    i2 = (i + n - 2) % n
                    i3 = i2 + n
                    i4 = i + n  - 1
                    self.Vertices2Facets.append([i1, i2, i3, i4])
                    
                self.Vertices2Facets.append(list(range(0, n)))
                self.Vertices2Facets.append(list(range(2*n - 1, n-1, -1)))
    
                un = np.zeros((2*n,3))
                for i in range(n):
                    un[i] = np.cross(v[(i + 1) % n, :] - v[i, :], v[-2, :] - v[i, :])
                    un[i+n] = (np.cross(v[-1, :] - v[i, :], v[(i + 1) % n, :] - v[i, :]))
                
                self.Normals = np.array(un) / np.linalg.norm(un, axis=1, keepdims=True)
                
        
                self.Edges = np.zeros((3*n, 2), dtype=int)
                for i in range(n):
                    self.Edges[i, :] = [i, (i+ 1) % n ]
                    self.Edges[i+n, :] = [i, n]
                    self.Edges[i+2*n, :] = [n+1, i]
                
                self.Facets = [[] for i in range(2*n)]
                for i in range(1, n+1):
                    i1 = i
                    i2 = n + 1 + (i % n)
                    i3 = - n - i
                    i2b = -i2 - n
                    i3b = -i3 + n
                    self.Facets[i-1] = [i1, i2, i3]
                    self.Facets[i+n-1] = [i1, i2b, i3b]
                
                self.Edges2Facets = np.zeros((len(self.Edges), 2), dtype=int)
                for i in range(len(self.Facets)):
                    edges = np.abs(self.Facets[i])
                    for j in range(len(edges)):
                        if self.Edges2Facets[edges[j]-1, 0] == 0:
                            self.Edges2Facets[edges[j]-1, 0] = i
                        else:
                            self.Edges2Facets[edges[j]-1, 1] = i
                            
                
            elif isinstance(domain_type, str) and "prism" in domain_type.lower():
                self.Dimension  = 3
                n = int(''.join(filter(str.isdigit, domain_type.lower())))
                self.Type = "Prism"+str(n)

                theta = np.linspace(0, 2 * np.pi, n + 1) + np.pi / n
                theta = theta[:-1]
                
                R = mult(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([[radius], [0]]))
                v = R.reshape(2, n).T
                
                v = np.vstack([np.hstack([v, -radius / 2 * np.ones((n, 1))]), np.hstack([v, radius / 2 * np.ones((n, 1))])])
                self.Vertices = v
            
                self.Vertices2Facets = [[] for i in range(2*n)]
                for i in range(0, n):
                    i1 = i
                    i2 = (i + n - 1) % n
                    self.Vertices2Facets[i] = [i1, i2, n]
                    self.Vertices2Facets[i + n] = [i1, n + 1, i2]
            
                self.Edges = np.zeros((3 * n, 2), dtype=int)
                for i in range(n):
                    i2 = (i+1) % n
                    self.Edges[i, :] = [i, i2]
                    self.Edges[i + n , :] = [i + n, i2 + n]
                    self.Edges[i + 2 * n , :] = [i, i + n]
            
                self.Facets = [[] for i in range(n+2)]
                for i in range(1, n + 1):
                    i1 = i
                    i2 = (i % n) + 1 + 2 * n
                    i3 = -((i - 1) % n) - 1 - n
                    i4 = -((i - 1) % n) - 1 - 2 * n
                    self.Facets[i-1] = [i1, i2, i3, i4]
            
                self.Facets[n] = list(range(1, n + 1))
                self.Facets[n + 1] = list(range(n + 1, 2 * n + 1))
            
                un = np.zeros((n+2,3))
                for i in range(n):
                    un[i,:] = np.cross(v[((i+1) % n), :] - v[i, :], v[n + i , :] - v[i, :])
                
                un[-2,:] = [0, 0, -1]
                un[-1,:] = [0, 0, 1]
                self.Normals = np.array(un) / np.linalg.norm(un, axis=1, keepdims=True)
            
                self.Edges2Facets = -np.ones((len(self.Edges), 2), dtype=int)
                for i in range(len(self.Facets)):
                    edges = np.abs(self.Facets[i])
                    for j in range(len(edges)):
                        if self.Edges2Facets[edges[j]-1, 0] < 0:
                            self.Edges2Facets[edges[j]-1, 0] = i
                        else:
                            self.Edges2Facets[edges[j]-1, 1] = i
                            
            # other interesting domains (TODO) : pyramid 8 (half diamond)
            
            self.NormalFan = NormalFan(self, length = lengthNormalFan, epsProj=epsProj)   


    def projection(self,rho):
        if self.Dimension == 0:
            return np.ones(rho.shape)
        
        elif self.Dimension == 1:
            rhoProj = rho.copy()
            mi, ma = np.min(self.Vertices), np.max(self.Vertices)
            rhoProj[rho < mi] = mi
            rhoProj[rho > ma] = ma
            return rhoProj
        
        elif self.Dimension == 2:
            return projection2D(rho, self)
        elif self.Dimension == 3:
            return projection3D(rho, self)

    def printInfo(self):
        print(f"Dimension : {self.Dimension}")
        print(f"Vertices ({self.Vertices.shape[0]}) :\n {self.Vertices}")
    
    def plot(self, vCenter = np.array([[0,0,0]]), 
             color = 'r', marker = 'o', alpha = 0.5, linewidth = 1, textFlag = True):
        ax = plt.gcf().gca()
        
        # to anticipate with the hierarchical vizualization
        pts = self.Vertices
        if self.Dimension == 0:
            if self.Vertices.shape[1] < 3:
                pts = np.hstack([pts,np.zeros((1,2))])
            pts = pts + vCenter
            if "3d" in str(type(ax)):
                ax.scatter(pts[0,0],pts[0,1], pts[0,2], color = color, marker = marker)
                if textFlag: ax.text(pts[0,0],pts[0,1], pts[0,2], "  v0", color = color[0:3])
            else:
                ax.scatter(pts[0,0],pts[0,1], color = color, marker = marker)
                if textFlag: ax.text(pts[0,0],pts[0,1], "  v0", color = color[0:3])
        elif self.Dimension == 1:
            
            if self.Vertices.shape[1] < 3:
                pts = np.hstack([pts,np.zeros((2,2))])
            pts = pts + vCenter
            if "3d" in str(type(ax)):
                ax.plot(pts[:,0],pts[:,1], pts[:,2], c = color, marker=marker)
                if textFlag: 
                    ax.text(pts[0,0],pts[0,1], pts[0,2], "  v0")
                    ax.text(pts[1,0],pts[1,1], pts[1,2], "  v1")
            else:
                ax.plot(pts[:,0],pts[:,1], c = color, marker=marker)
                if textFlag: 
                    ax.text(pts[0,0],pts[0,1], "v0\n", color = color[0:3])
                    ax.text(pts[1,0],pts[1,1], "v1\n", color = color[0:3])
                    
        elif self.Dimension == 2:
            nb = self.Vertices.shape[0]
            if self.Vertices.shape[1] < 3:
                pts = np.hstack([pts, np.zeros((nb,1))])
            pts = pts + vCenter    
            if "3d" in str(type(ax)):
                pts = np.vstack([pts, np.mean(pts,axis = 0)])
                meshSurf = [[i, (i+1) % nb, nb] for i in range(nb) ]
                ax.plot_trisurf(pts[:,0], pts[:,1], pts[:,2],
                            triangles=meshSurf, linewidth=linewidth,  color = color, alpha = alpha)
                e = np.vstack([pts[:-1,:],pts[0,:]])
                line_collection = Line3DCollection([e], colors='k', linewidths=linewidth)
                ax.add_collection(line_collection)
                ax.plot(pts[:-1,0],pts[:-1,1], pts[:-1,2], color = color[0:3], marker = marker)
                if textFlag: 
                    for i in range(pts.shape[0]):
                        ax.text(pts[i,0],pts[i,1], pts[i,2], "  v"+str(i), color = color[0:3])
            else:
                ax.plot(pts[:,0],pts[:,1], color + marker)
                ax.add_collection(PatchCollection([Polygon(self.Vertices)], alpha=alpha, color = color))
                if textFlag: 
                    for i in range(pts.shape[0]):
                        ax.text(pts[i,0],pts[i,1], "  v"+str(i), color = color[0:3])
        
        elif self.Dimension == 3:
            if "3d" not in str(type(ax)):
                ax = plt.gcf().add_subplot(projection='3d')
                 
            vCentered=self.Vertices + vCenter
            
            for facets in self.Facets:
                pointsFacets = []
                for edges in facets:
                    if edges>0 : pointsFacets.append(self.Edges[edges-1][0])
                    else : pointsFacets.append(self.Edges[abs(edges)-1][1])
                
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
                                    
                    tri = Triangulation(ptRef[:,0],ptRef[:,1])
                    
                    pointsTriLocal = Ref2Facet(ptRef)
                    p3dc = ax.plot_trisurf(pointsTriLocal[0], pointsTriLocal[1], pointsTriLocal[2],
                                    triangles=tri.triangles,  color = color, alpha = alpha)
            nn = self.Vertices.shape[0]
            e = [np.vstack([vCentered[self.Edges[i][0]], 
                                      vCentered[self.Edges[i][1]]]) for i in range(len(self.Edges)) ]
            line_collection = Line3DCollection(e, colors='k', linewidths=linewidth)
            ax.add_collection(line_collection)
            ax.plot(vCentered[:,0],vCentered[:,1], vCentered[:,2], ''+marker, color = color)
            if textFlag: 
                for i in range(pts.shape[0]):
                    ax.text(pts[i,0],pts[i,1], pts[i,2], "  v"+str(i), color = color[0:3])
                    
        plt.axis("equal")

        
    def plotProjection(self, rho = 8*(np.random.rand(50,3)-0.5), alpha = 0):
        dim = self.Dimension
        rhoProj = self.projection(rho[:,0:dim])
        arcs = [np.vstack([rho[i,0:dim], rhoProj[i,0:dim]]) for i in range(rho.shape[0])]
        ax = plt.gca()
        if self.Dimension == 3 :
            if "3d" not in str(type(ax)):
                ax = plt.gcf().add_subplot(projection='3d')
            self.plot(alpha = alpha)
            line_collection = Line3DCollection(arcs, linestyle = '--')
            ax.plot(rho[:,0],rho[:,1],rho[:,2], 'bo')
            ax.plot(rhoProj[:,0],rhoProj[:,1],rhoProj[:,2] , 'r*')
        else:
            self.NormalFan.plot()
            line_collection = LineCollection(arcs, linestyle = '--')
            ax.plot(rho[:,0],rho[:,1], 'bo')
            ax.plot(rhoProj[:,0],rhoProj[:,1], 'r*')
        ax.add_collection(line_collection)

'''
Normal fan (for projection)
'''

class NormalFan:
    def __init__(self, domain, length = 1, epsProj = 1e-5):
        self.EpsProj = epsProj
        dProj = deepcopy(domain)
        dProj.Vertices = dProj.Vertices * (1-epsProj)
        self.Length = length
        if domain.Dimension == 2:
            rectangles, triangles = normalFan2D(dProj, length)
            self.ConesEdges = rectangles
            self.ConesVertices = triangles
            self.ConesFacets = []
            self.Polygon = Simple2Dpolygon(dProj.Vertices)
        elif domain.Dimension == 3:
            prismFacet, roofsEdges, conesVertices = normalFan3D(dProj, length)
            self.ConesEdges = roofsEdges
            self.ConesVertices = conesVertices
            self.ConesFacets = prismFacet
            pf = np.vstack([dProj.Vertices[np.abs(dProj.Facets[i][0])-1, :] for i in range(len(dProj.Facets))])
            self.Polygon = Simple3Dpolyedron(dProj.Vertices, normals = dProj.Normals, pointsFacets= pf)
         
    def plot(self, alpha=0.1):
        plt.figure()
        if len(self.ConesFacets)!=0:
            plt.gcf().add_subplot(projection='3d')

        for i in range(len(self.ConesFacets)):
            self.ConesFacets[i].plot(color = [0,0,1], alpha = alpha)
        
        for coneE in self.ConesEdges:
            coneE.plot(color = [0,1,0], alpha = alpha)
            
        for coneV in self.ConesVertices:
            coneV.plot(color = [1,0,0], alpha = alpha)
        plt.gca().set_aspect('equal', 'box')
        plt.axis('off')
        plt.show()
        
    
def normalFan2D(domain, length = 1):
    rectangles = []
    triangles = []
    n = domain.Vertices.shape[0]
    
    # 1) rectangle
    for i in range(n):
        p1 = domain.Vertices[i]
        p2 = domain.Vertices[(i+1)%n]
        d = p2-p1
        un = np.array([d[1], -d[0]])
        p3 = p2 + un*length
        p4 = p1 + un*length
        rectangles.append(Simple2Dpolygon(np.array([p1,p2,p3,p4])))
    
    # 2) triangle
    for  i in range(n):
        p1 = domain.Vertices[i]
        p2 = rectangles[i].Vertices[3]
        p3 = rectangles[(i-1)%n].Vertices[2]
        triangles.append(Simple2Dpolygon([p1,p2,p3]))
        
    return rectangles, triangles


def normalFan3D(domain, length = 1):
    prismFacet = []
    roofsEdges = []
    conesVertices = []
    f = domain.Facets
    e = domain.Edges
    v = domain.Vertices
    un = domain.Normals
    nf, ne, nv = len(f), len(e), len(v)
    
    # 1) Facets
    for i in range(nf):
        nodesF = [e[abs(j)-1][0]*(j>0) + e[abs(j)-1][1]*(j<0) for j in f[i]]
        base1=v[nodesF]
        prismFacet.append(prism3D(base1, un[i]*length))
        
    # 2) Edges
    for i in range(ne):
        nodesE = e[i]
        un1 = un[domain.Edges2Facets[i][0]]
        un2 = un[domain.Edges2Facets[i][1]]
        ve1 = v[nodesE[0]]
        ve2 = v[nodesE[1]]
        base1=np.vstack([ve1, ve1 + un1 * length, ve1+ un2*length])
        base2=np.vstack([ve2, ve2 + un1 * length, ve2+ un2*length])
        roofsEdges.append(prism3D(base1, base2))
        
    # 3) Vertices
    for i in range(nv):
        v1 = v[i]
        base = np.array([un[j]*length + v1 for j in domain.Vertices2Facets[i]])
        conesVertices.append(pyramid3D(base, [v1]))
    
    return prismFacet, roofsEdges, conesVertices

'''
Utilities for defining the normal fan
'''

class Simple2Dpolygon:
    def __init__(self, vertices): # should be in trigonometric order
        self.Vertices = np.array(vertices)
        vn =  np.vstack([self.Vertices[1:,:]-self.Vertices[:-1,:],self.Vertices[0,:]-self.Vertices[-1,:]])
        R = np.array([[0, 1],[-1, 0]])
        self.Normals = vn @ R.T
        vCenter = np.mean(self.Vertices, axis = 0)
        self.Area = self.computeArea(vCenter)
        # check orientation
        for i in range(len(self.Normals)):
            self.Normals[i] = - self.Normals[i] * np.sum( self.Normals[i] * (vCenter-self.Vertices[i]))
            self.Normals[i] = self.Normals[i] / np.linalg.norm(self.Normals[i])
        
    def computeArea(self, pTest):
        # pTest : Nx2 array
        area = []
        n= self.Vertices.shape[0]
        pTest = pTest.reshape(-1,2)
        for i in range(n):
            p1 = self.Vertices[i]
            p2 = self.Vertices[(i+1) % n]
            area.append(np.abs(np.cross(p1-pTest,p2-pTest, axis = 1)))
        area = np.sort(np.array(area), axis = 0) # slow but prevents roundoff errors
        area=np.sum(area, axis = 0)
        return area / 2
        
    def isInsideOld(self, pTest, tol = 1e-10): # slow
        vol = self.computeArea(pTest)
        return vol <= (self.Volume * (1+tol)) 
    
    def isInside(self, pTest, inf = None, tol = 1e-12): # recommended
        pTest = (pTest.T).reshape(1,2,-1)
        vec = pTest - self.Vertices.reshape(-1,2,1)
        un = self.Normals.reshape(-1,2,1)
        if inf is not None:
            vec = np.delete(vec, inf, axis = 0)
            un =  np.delete(un, inf, axis = 0)
        dot = np.sum(np.sum(vec*un , axis = 1) > -tol, axis = 0).flatten()
        return dot == 0
        
    def plot(self, color = 'r', alpha = 0.5):
        sequence = np.concatenate([np.arange(0,self.Vertices.shape[0]),[0]], axis = 0)
        plt.plot(self.Vertices[sequence,0],self.Vertices[sequence,1],'-ko')
        polygon = Polygon(self.Vertices, closed=True, color = color, alpha = alpha)
        plt.gca().add_patch(polygon)
 
        

class Simple3Dpolyedron:
    def __init__(self, vertices, normals = None, meshSurf = None, pointsFacets = None, vol = None):
        self.Vertices = np.array(vertices)
        self.MeshSurf = meshSurf
        self.Normals = normals
        self.PointsFacets = pointsFacets
        self.Volume = vol
    
    def plot(self, color = [0,0,0], alpha = 0.5):
        if "3d" not in str(type(plt.gca())): plt.close() ; plt.figure();  plt.gcf().add_subplot(projection='3d')
        pts = self.Vertices
        plt.gca().plot_trisurf(pts[:,0], pts[:,1], pts[:,2],
                        triangles=self.MeshSurf, linewidth=0.5,  color = color + [alpha])

    def computeVol(self, pTest):
        # pTest : Nx2 array
        vol = []
        n = len(self.MeshSurf)
        pTest = pTest.reshape(-1,3)
        for i in range(n):
            p1 = self.Vertices[self.MeshSurf[i][0]]
            p2 = self.Vertices[self.MeshSurf[i][1]]
            p3 = self.Vertices[self.MeshSurf[i][2]]
            vol.append(np.abs(np.sum(np.cross(p1-pTest, p2-pTest, axis = 1) * (p3-pTest), axis = 1)))
        vol = np.sort(np.array(vol), axis = 0) # slow but prevents roundoff errors
        vol=np.sum(vol, axis = 0)
        return vol / 6
    
    def isInsideOld(self, pTest, tol = 1e-10): # slow
        vol = self.computeVol(pTest)
        return vol <= (self.Volume * (1+tol)) 
    
    def isInside(self, pTest, inf = None, tol = 1e-12): # recommended
        pTest = (pTest.T).reshape(1,3,-1)
        vec = pTest - self.PointsFacets.reshape(-1,3,1)
        un = self.Normals.reshape(-1,3,1)
        if inf is not None:
            vec = np.delete(vec, inf, axis = 0)
            un =  np.delete(un, inf, axis = 0)
        dot = np.sum(np.sum(vec*un , axis = 1) > -tol, axis = 0).flatten()
        return dot == 0


def prism3D(base1, base2orExtrude):
    # the vertices in "base" should be ordered and consistent
    nb = base1.shape[0]
    if base2orExtrude.size == 3: base2 = base1 + base2orExtrude.reshape(1,3)
    else : base2 = base2orExtrude
    base1Center = np.mean(base1,axis = 0, keepdims=True)
    base2Center = np.mean(base2,axis = 0, keepdims=True)
    vertices = np.concatenate([base1, base1Center,base2,base2Center], axis = 0)
    
    meshSurf1 = [[i, (i+1) % nb, nb] for i in range(nb) ]
    meshSurf2 = [[nb+1+ i, nb+1 + ((i+1) % nb), 2*nb+1] for i in range(nb) ]
    meshSurfLat1 = [[i, (i+1) % nb, nb+1 + i,] for i in range(nb) ]
    meshSurfLat2 = [[nb+1+ i, nb+1 + ((i+1) % nb) , (i+1) % nb] for i in range(nb) ]

    meshSurf = meshSurf1 + meshSurf2+ meshSurfLat1 + meshSurfLat2
    
    prism = Simple3Dpolyedron(vertices, meshSurf = meshSurf)
    prism.Volume = prism.computeVol(np.mean(vertices, axis = 0))
    
    ## for efficient isInside test
    unBase1 = np.cross(base1[1]-base1[0], base1[2]-base1[0])
    unBase2 = np.cross(base2[1]-base2[0], base2[2]-base2[0])
    unBase1 = - unBase1 * np.sum( unBase1 * (base2[0]-base1[0]))
    unBase2 = - unBase2 * np.sum( unBase2 * (base1[0]-base2[0]))
    unBase1 = unBase1 / np.linalg.norm(unBase1)
    unBase2 = unBase2 / np.linalg.norm(unBase2)
    pfBase1, pfBase2 = base1[0],  base2[0]
    unSide = [np.cross(base2[i]-base1[i], base1[(i+1)%nb]-base1[i]) for i in range(nb) ]
    pfSide = base1
    vCenter = np.mean(vertices, axis = 0)
    # check orientations
    for i in range(len(unSide)):
        unSide[i] = - unSide[i] * np.sum( unSide[i] * (vCenter-base1[i]))
        unSide[i] = unSide[i] / np.linalg.norm(unSide[i])
    prism.Normals = np.vstack([unBase1, *unSide, unBase2])
    prism.PointsFacets = np.vstack([pfBase1, *pfSide, pfBase2])
    return prism


def pyramid3D(base, top):
    # the vertices in "base" should be ordered (not necessarily in trigonometric order)
    nb = base.shape[0]
    baseCenter = np.mean(base,axis = 0, keepdims=True)
    vertices = np.concatenate([base, baseCenter, top], axis = 0)
    
    meshSurfBase = [[i, (i+1) % nb, nb] for i in range(nb) ]
    meshSurfLat = [[i, (i+1) % nb, nb+1] for i in range(nb) ]
    meshSurf = meshSurfBase + meshSurfLat
    
    pyramid = Simple3Dpolyedron(vertices, meshSurf = meshSurf)
    pyramid.Volume = pyramid.computeVol(np.mean(vertices, axis = 0))
    
    ## for efficient isInside test
    unBase = np.cross(base[1]-base[0], base[2]-base[0])
    unBase = - unBase * np.sum( unBase * (top-base[0]))
    unBase = unBase / np.linalg.norm(unBase)
    pfBase = base[0]
    unTop = [np.cross(top-base[(i+1)%nb], base[i]-base[(i+1)%nb]) for i in range(nb) ]
    pfTop = base
    # check orientations
    vCenter = np.mean(vertices, axis = 0)
    for i in range(len(unTop)):
        unTop[i] = - unTop[i] * np.sum( unTop[i] * (vCenter-base[i]))
        unTop[i] = unTop[i] / np.linalg.norm(unTop[i])
    
    pyramid.Normals = np.vstack([*unTop, unBase])
    pyramid.PointsFacets =  np.vstack([pfTop, pfBase])
    
    return pyramid

'''
2D projection routines
'''

def proj(vector, q):
    p0 = vector[0, :]
    p1 = vector[1, :]
    q = q.reshape(1,1,-1,2)
    
    a = np.array([
        -q[0, 0, :, 0] * (p1[0] - p0[0]) - q[0, 0, :, 1] * (p1[1] - p0[1]),
        np.repeat(-p0[1] * (p1[0] - p0[0]) + p0[0] * (p1[1] - p0[1]), q.shape[2])
    ])
    
    b = np.array([
        [p1[0] - p0[0], p1[1] - p0[1]],
        [p0[1] - p1[1], p1[0] - p0[0]]
    ])
    
    detb = b[0, 0] * b[1, 1] - b[0, 1] * b[1, 0]
    invb = np.array([[b[1, 1], -b[0, 1]], [-b[1, 0], b[0, 0]]]) / detb
    
    ProjPoint = -mult(invb, a).T

    return ProjPoint

def projection2D(rho, domain):
    """
    Returns projected values onto a convex 2D polygon.
    """
    
    rhoProj = rho.copy()
    dProj = domain.NormalFan.Polygon
    nv = dProj.Vertices.shape[0]
    nr = rho.shape[0]
    indToDo = np.full(nr, True)

    # 0) remove points inside polygon
    
    inPoly = dProj.isInside(rhoProj)
    index = np.full(nr, False)
    index[indToDo] = inPoly
    indToDo = indToDo & ~index
    
    # 1) projection on vertices (triangles in the normal fan)

    for i in range(nv):
        triangle = domain.NormalFan.ConesVertices[i]
        inTri= triangle.isInside(rhoProj[indToDo,:], inf = 1 )
        index = np.full(nr, False)
        index[indToDo] = inTri
        indToDo = indToDo & ~index
        rhoProj[index, :] = dProj.Vertices[i, :]

    # 2) projection on edges (rectangles in the normal fan)
    for i in range(nv):
        rectangle = domain.NormalFan.ConesEdges[i]
        inRect= rectangle.isInside(rhoProj[indToDo,:], inf = 2)
        edge = np.vstack([dProj.Vertices[i],dProj.Vertices[(i+1)%nv]])
        index = np.full(nr, False)
        index[indToDo] = inRect
        indToDo = indToDo & ~index
        projpoints=proj(edge, rhoProj[index,:])
        rhoProj[index,:]= projpoints
    #indToDo[indToDo] =  ~ Simple2Dpolygon(domain.Vertices).isInside(rhoProj[indToDo,:])
    #plt.plot(indToDo)
    return rhoProj

'''
3D projection routines
'''

def projEdges(vec_dir, p, q):
    q = np.reshape(q.T, (1,3,-1))
    p = np.reshape(p, (1,3,1))
    vec_dir =np.reshape( vec_dir / np.linalg.norm(vec_dir, 2),(3,1,1))
    proj_points = p + mult(q - p, vec_dir)*t(vec_dir)
    return proj_points.transpose(2,1,0).reshape(-1,3)

def projPlane(n, p, q):
    """
    Project q onto the plane defined by normal vector n and point p.
    """
    q = np.reshape(q.T, (1,3,-1))
    p = np.reshape(p, (1,3,1))
    n = np.reshape(n, (1,3,1))
    proj_points = q - mult(q - p, n.reshape(-1,1)) * n
    return proj_points.transpose(2,1,0).reshape(-1,3)




def projection3D(rho, domain):
    """
    Returns projected values onto a convex 3D polyedron
    """
    dProj = domain.NormalFan.Polygon
    nr = rho.shape[0]
    indToDo = np.full(nr, True)
    rhoProj = rho.copy()
    #eps = domain.NormalFan.EpsProj
    
    # 0) remove points inside polyedron
    
    inPoly = dProj.isInside(rhoProj)
    index = np.full(nr, False)
    index[indToDo] = inPoly
    indToDo = indToDo & ~index

    # 1) projection on vertices (cones in the normal fan)
    cv = domain.NormalFan.ConesVertices
    for i in range(len(cv)):
        inTri= cv[i].isInside(rhoProj[indToDo,:], inf = -1)
        index = np.full(nr, False)
        index[indToDo] = inTri
        indToDo = indToDo & ~index
        rhoProj[index, :] = dProj.Vertices[i, :]

    # 2) projection on edges (roofs in the normal fan)
    ce = domain.NormalFan.ConesEdges
    for i in range(len(ce)):
        inRect= ce[i].isInside(rhoProj[indToDo,:], inf = 2)
        index = np.full(nr, False)
        index[indToDo] = inRect
        indToDo = indToDo & ~index
        p = dProj.Vertices[domain.Edges[i][0],:]
        vec_dir = dProj.Vertices[domain.Edges[i][1],:] - p
        rhoProj[index,:]= projEdges(vec_dir, p, rhoProj[index,:])
        
     # 3) projection on facets (prisms in the normal fan)
    cf = domain.NormalFan.ConesFacets
    for i in range(len(cf)):
        inPrism= cf[i].isInside(rhoProj[indToDo,:], inf = -1)
        index = np.full(nr, False)
        index[indToDo] = inPrism
        indToDo = indToDo & ~index
        un = domain.Normals[i]
        edg = domain.Edges[abs(domain.Facets[i][0])-1]
        p = dProj.Vertices[edg[0],:]
        vec_dir = dProj.Vertices[edg[1],:] - p
        rhoProj[index,:]= projPlane(un, p,  rhoProj[index,:])
    
    # debug
    #indToDo[indToDo] =  ~ pyramid3D(domain.Vertices[:-1,:], [domain.Vertices[-1,:]]).isInside(rhoProj[indToDo,:])
    #domain.NormalFan.plot()
    #plt.gca().scatter(rhoProj[indToDo,0],rhoProj[indToDo,1],rhoProj[indToDo,2],s = 100)
    #plt.plot(indToDo)
    return rhoProj



#Domain(3).plot()
