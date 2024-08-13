from netgen.geom2d import SplineGeometry
from ngsolve import *

def poleMotor(Npp = 2, Rrmin=0.02, Rrmax = 0.0495, Rsmin = 0.0505 ,Rsmax = 0.07, maxh=0.1):
    geo = SplineGeometry()
    pref = [(1,0),(1,tan(pi/(2*Npp))),(cos(2*pi/(2*Npp)),sin(2*pi/(2*Npp)))]
    P1= [ geo.AppendPoint(pnt[0]*Rrmin, pnt[1]*Rrmin) for pnt in pref ]
    P2= [ geo.AppendPoint(pnt[0]*Rrmax, pnt[1]*Rrmax) for pnt in pref ]
    P3= [ geo.AppendPoint(pnt[0]*Rsmin, pnt[1]*Rsmin) for pnt in pref ]
    P4= [ geo.AppendPoint(pnt[0]*Rsmax, pnt[1]*Rsmax) for pnt in pref ]
    
    geo.Append(['spline3',P1[0],P1[1],P1[2]], leftdomain=0, rightdomain=1, bc="a0", maxh = maxh)
    geo.Append(['spline3',P2[0],P2[1],P2[2]], leftdomain=1, rightdomain=2, bc="e1", maxh = maxh/3)
    geo.Append(['spline3',P3[0],P3[1],P3[2]], leftdomain=2, rightdomain=3, bc="e2", maxh = maxh/3)
    geo.Append(['spline3',P4[0],P4[1],P4[2]], leftdomain=3, rightdomain=0, bc="a0", maxh = maxh)
    
    y1 = geo.Append(['line',P1[0],P2[0]], leftdomain=1, rightdomain=0, bc="antiPeriodic")
    geo.Append(['line',P1[2],P2[2]], leftdomain=0, rightdomain=1, bc="antiPeriodic", copy = y1)
    y2 = geo.Append(['line',P2[0],P3[0]], leftdomain=2, rightdomain=0, bc="antiPeriodic")
    geo.Append(['line',P2[2],P3[2]], leftdomain=0, rightdomain=2, bc="antiPeriodic", copy = y2)
    y3 = geo.Append(['line',P3[0],P4[0]], leftdomain=3, rightdomain=0, bc="antiPeriodic")
    geo.Append(['line',P3[2],P4[2]], leftdomain=0, rightdomain=3, bc="antiPeriodic", copy = y3)

    geo.SetMaterial(1,"Rotor"); geo.SetDomainMaxH(1,maxh/2)
    geo.SetMaterial(2,"Airgap") ; geo.SetDomainMaxH(2,maxh/3)
    geo.SetMaterial(3,"Stator")
    mesh=Mesh(geo.GenerateMesh(maxh=maxh, grading = 0.05)).Curve(3)
    return mesh