from .utils import mult, t

from copy import copy, deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class VertexFunction:
    '''
    VertexFunction Class
    
     To create a VertexFunction object, type
     obj = VertexFunction(expression,derivative,label,dimInput,dimOuput)
    
     where 'expression' and 'derivative' are function handles
           'label' is a string identifier (default "")
           'dimInput' and 'dimOutput' are numerics (default 1)
    
     Following operators are overloaded : + - * / **
    
     A vertexFunction is a function depending on a variable a (scalar or
     vector), and that will combined with others in an interpolation that
     depends on another variable x.
    
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
    def __init__(self, label, f, dfdu, dimInput = 1 , dimOutput = 1):
        
        # Checks
        assert isinstance(label, str), "label should be a string"
        assert isinstance(f, type(lambda x : 1)), "expr should be a function"
        assert isinstance(dfdu, type(lambda x : 1)), "derivative should be a function"
        assert isinstance(dimInput, int) and dimInput>= 1, "dimInput should be an integer >= 1"
        assert np.array(dimOutput).reshape(-1).size <=2, "Only vector and matrices operations on VertexFunctions are supported"
        #assert np.array(dimInput).reshape(-1).size == 1 or np.array(dimOutput).reshape(-1).size == 1, "Matrix operations are supported only when dimInput = 1"
        
        self.Label = label
        self.Expression = f
        self.Derivative = dfdu
        self.DimInput = dimInput
        self.DimOutput = dimOutput
        self.FlagParExp = False
        self.FlagParMult = False
    def eval(self, u):
        return self.Expression(u)
    
    def evald(self, u):
        return self.Derivative(u)
    
    def __add__(self, other):
        result = deepcopy(self)
        result.FlagParExp = True
        result.FlagParMult = True
        if isinstance(other, (int, float)) :
            f = deepcopy(self.Expression)
            result.Expression = lambda u : f(u) + other
            result.Label = self.Label + " + "+ str(other)
        
        elif isinstance(other, VertexFunction) :
            assert self.DimInput == other.DimInput, "VertexFunctions should have the same InputDimension to be added together"
            dimOut2, dimOut1 = np.array([self.DimOutput]).squeeze(), np.array([other.DimOutput]).squeeze()
            try: np.broadcast(np.zeros(dimOut1), np.zeros(dimOut2))
            except : raise Exception("VertexFunctions should be broadcastable to be added together") 
            f, g= deepcopy(self.Expression), deepcopy(other.Expression)
            dfdu, dgdu =  deepcopy(self.Derivative), deepcopy(other.Derivative)
            result.Expression = lambda u : f(u) + g(u)
            result.Derivative = lambda u : dfdu(u) + dgdu(u)
            result.DimOutput = np.broadcast(np.zeros(dimOut1), np.zeros(dimOut2)).shape
            result.Label = self.Label + " + "+ other.Label
        return result
    
    def __radd__(self, other):
        result = deepcopy(self)
        result.FlagParExp = True
        result.FlagParMult = True
        if isinstance(other, (int, float)) :
            f = deepcopy(self.Expression)
            dimOut2, dimOut1 = np.array([self.DimOutput]).squeeze(), np.array([other.DimOutput]).squeeze()
            try: np.broadcast(np.zeros(dimOut1), np.zeros(dimOut2))
            except : raise Exception("VertexFunctions should be broadcastable to be added together") 
            result.Expression = lambda u : other + f(u)
            result.Label = str(other) + " + " + self.Label
            result.DimOutput = np.broadcast(np.zeros(dimOut1), np.zeros(dimOut2)).shape
        return result
    
    def __sub__(self, other):
        result = deepcopy(self)
        result.FlagParExp = True
        result.FlagParMult = True
        if isinstance(other, (int, float)) :
            f = deepcopy(self.Expression)
            result.Expression = lambda u : f(u) - other
            result.Label = self.Label + " - "+ str(other)
        
        elif isinstance(other, VertexFunction) :
            assert self.DimInput == other.DimInput, "VertexFunctions should have the same InputDimension to be substracted together"
            dimOut1, dimOut2 = np.array([self.DimOutput]).squeeze(), np.array([other.DimOutput]).squeeze()
            try: np.broadcast(np.zeros(dimOut1), np.zeros(dimOut2))
            except : raise Exception("VertexFunctions should be broadcastable to be substracted together") 
            f, g= deepcopy(self.Expression), deepcopy(other.Expression)
            dfdu, dgdu =  deepcopy(self.Derivative), deepcopy(other.Derivative)
            result.Expression = lambda u : f(u) - g(u)
            result.Derivative = lambda u : dfdu(u) - dgdu(u)
            result.DimOutput = np.broadcast(np.zeros(dimOut1), np.zeros(dimOut2)).shape
            result.Label = self.Label + " - "+ other.Label
        return result
    
    def __rsub__(self, other):
        result = deepcopy(self)
        result.FlagParExp = True
        result.FlagParMult = True
        if isinstance(other, (int, float)) :
            f = deepcopy(self.Expression)
            result.Expression = lambda u : other - f(u)
            result.Label = str(other) + " - " + self.Label
        return result
    
    def __neg__(self):
        result = deepcopy(self)
        f = deepcopy(self.Expression)
        dfdu =  deepcopy(self.Derivative)
        result.Expression = lambda u : -f(u)
        result.Derivative = lambda u : -dfdu(u)
        
        if result.FlagParMult: result.Label = "-(" + self.Label + ")"
        else :  result.Label = "-" + self.Label
        result.FlagParExp = True
        result.FlagParMult = False
        return result
    
    def __mul__(self, other): # term wise multiplication
        result = deepcopy(self)
        dimOut1 = np.array([self.DimOutput]).squeeze()
        if isinstance(other, (int, float)) :
            f = deepcopy(self.Expression)
            dfdu =  deepcopy(self.Derivative)
            result.Expression = lambda u : f(u) * other
            result.Derivative = lambda u : dfdu(u) * other
            
            if result.FlagParMult: result.Label =  "(" + self.Label + ") * " + str(other)
            else :  result.Label = self.Label+ " * " + str(other) 
        
        elif isinstance(other, VertexFunction) :
            assert self.DimInput == other.DimInput, "VertexFunctions should have the same InputDimension to be multiplied together"
            # assert self.DimOutput == other.DimOutput, "Two VertexFunctions should have the same OutputDimension to be multiplied together" # may be ok because of broadcasting
            dimOut2 = np.array([other.DimOutput]).squeeze()
            try: np.broadcast(np.zeros(dimOut1), np.zeros(dimOut2))
            except : raise Exception("VertexFunctions should be broadcastable to be multiplied together") 
                
            f, g= deepcopy(self.Expression), deepcopy(other.Expression)
            dfdu, dgdu =  deepcopy(self.Derivative), deepcopy(other.Derivative)
            result.Expression = lambda u : f(u) * g(u)
            result.Derivative = lambda u : dfdu(u)*g(u) + dgdu(u)*f(u)

            if result.FlagParMult and other.FlagParMult : result.Label = "(" + self.Label + ") * (" + other.Label + ")"
            elif result.FlagParMult and not other.FlagParMult : result.Label = "(" + self.Label + ") * " + other.Label
            elif not result.FlagParMult and other.FlagParMult : result.Label =  self.Label + " * (" + other.Label + ")"
            else :  result.Label =  self.Label + " * " + other.Label
            result.DimOutput = np.broadcast(np.zeros(dimOut1), np.zeros(dimOut2)).shape
        result.FlagParExp = True
        result.FlagParMult = False
        return result
    
    def __rmul__(self, other):
        result = deepcopy(self)
        if isinstance(other, (int, float)) :
            f = deepcopy(self.Expression)
            dfdu =  deepcopy(self.Derivative)
            result.Expression = lambda u : other * f(u)
            result.Derivative = lambda u : other * dfdu(u)
            
            if result.FlagParMult: result.Label =  str(other) + "* (" + self.Label + ")"
            else :  result.Label =  str(other) + " * " + self.Label
        result.FlagParExp = True
        result.FlagParMult = False
        return result
    
    
    def __matmul__(self, other): # matrix multiplication
        result = deepcopy(self)
        if isinstance(other, (int, float)) :
            result = self * other
        
        elif isinstance(other, type(np.array(0))) :
            dimOut1, dimOut2 = np.array([self.DimOutput]).reshape(-1), np.array(other.shape).reshape(-1)
            try : assert dimOut1[1] == dimOut2[0] and len(dimOut1)<=2 and len(dimOut2)<=2, "Output dimensions of VertexFunctions don't match to perform matrix-multiplication"
            except : raise Exception("Output dimensions of VertexFunctions don't match to perform matrix-multiplication") 
            
            f = deepcopy(self.Expression)
            dfdu =  deepcopy(self.Derivative)
            result.Expression = lambda u : mult(f(u), other)
            result.Derivative = lambda u : mult(other.T, dfdu(u))            
            if len(dimOut2) == 1 : result.DimOutput = (dimOut1[0], )
            elif len(dimOut2) == 2 : result.DimOutput = (dimOut1[0], dimOut2[1])
            else : raise Exception("Output dimension higher than 2 are not supported") 
            
            if result.FlagParMult and other.FlagParMult : result.Label = "(" + self.Label + ") @ (" + str(other) + ")"
            elif result.FlagParMult and not other.FlagParMult : result.Label = "(" + self.Label + ") @ " + str(other)
            elif not result.FlagParMult and other.FlagParMult : result.Label =  self.Label + " @ (" + str(other) + ")"
            else :  result.Label =  self.Label + " @ " + str(other)
        
        elif isinstance(other, VertexFunction) :
            assert self.DimInput == other.DimInput, "VertexFunctions should have the same InputDimension to perform matrix-multiplication"
            # assert self.DimOutput == other.DimOutput, "Two VertexFunctions should have the same OutputDimension to be multiplied together" # may be ok because of broadcasting
            dimOut1, dimOut2 = np.array([self.DimOutput]).reshape(-1), np.array([other.DimOutput]).reshape(-1)
            try : assert dimOut1[1] == dimOut2[0] and len(dimOut1)<=2 and len(dimOut2)<=2, "Output dimensions of VertexFunctions don't match to perform matrix-multiplication"
            except : raise Exception("Output dimensions of VertexFunctions don't match to perform matrix-multiplication") 
                
            f, g= deepcopy(self.Expression), deepcopy(other.Expression)
            dfdu, dgdu =  deepcopy(self.Derivative), deepcopy(other.Derivative)
            result.Expression = lambda u : mult(f(u) , g(u))
            result.Derivative = lambda u : mult(f(u), dgdu(u)) + mult(dfdu(u), g(u))
            if len(dimOut2) == 1 : result.DimOutput = (dimOut1[0], )
            elif len(dimOut2) == 2 : result.DimOutput = (dimOut1[0], dimOut2[1])
            else : raise Exception("Output dimension higher than 2 are not supported") 
            
            if result.FlagParMult and other.FlagParMult : result.Label = "(" + self.Label + ") @ (" + other.Label + ")"
            elif result.FlagParMult and not other.FlagParMult : result.Label = "(" + self.Label + ") @ " + other.Label
            elif not result.FlagParMult and other.FlagParMult : result.Label =  self.Label + " @ (" + other.Label + ")"
            else :  result.Label =  self.Label + " @ " + other.Label
        result.FlagParExp = True
        result.FlagParMult = False
        return result
    
    def __rmatmul__(self, other): # matrix multiplication
        result = deepcopy(self)
        if isinstance(other, (int, float)) :
            result = other * self
        
        elif isinstance(other, type(np.array(0))) :
            dimOut2, dimOut1 = np.array([self.DimOutput]).reshape(-1), np.array(other.shape).reshape(-1)
            try : assert dimOut1[1] == dimOut2[0] and len(dimOut1)<=2 and len(dimOut2)<=2, "Output dimensions of VertexFunctions don't match to perform matrix-multiplication"
            except : raise Exception("Output dimensions of VertexFunctions don't match to perform matrix-multiplication") 
            
            f = deepcopy(self.Expression)
            dfdu =  deepcopy(self.Derivative)
            result.Expression = lambda u : mult(other, f(u))
            result.Derivative = lambda u : mult(other, dfdu(u))
            
            if len(dimOut2) == 1 : result.DimOutput = (dimOut1[0], )
            elif len(dimOut2) == 2 : result.DimOutput = (dimOut1[0], dimOut2[1])
            else : raise Exception("Output dimension higher than 2 are not supported") 
            
            if result.FlagParMult: result.Label = "(" + str(other)  + ") @ (" + self.Label + ")"
            else :  result.Label =   str(other) + " @ " +self.Label
        result.FlagParExp = True
        result.FlagParMult = False
        return result
    
    def t(self):
        result = deepcopy(self)
        dimOut1 = np.array([self.DimOutput]).reshape(-1)
        if dimOut1.size == 1:
            result.DimOutput = (1, dimOut1[0])
        else:
            result.DimOutput = (dimOut1[1], dimOut1[0])
        f = deepcopy(self.Expression)
        dfdu =  deepcopy(self.Derivative)
        result.Expression = lambda u : t(f(u))
        result.Derivative = lambda u : t(dfdu(u))
        if result.FlagParMult : result.Label = "(" + self.Label + ")^t"
        else :  result.Label =  self.Label + "^t"
        result.FlagParExp = True
        result.FlagParMult = False
        return result
    
    
    def innerProduct(self,other):
        result = deepcopy(self)
        f = deepcopy(self.Expression)
        dfdu =  deepcopy(self.Derivative)
       
        if isinstance(other, (int, float) ) :
            dimOut1, dimOut2 = np.array([self.DimOutput]).reshape(-1),np.array([1])
            result = self * other
                
        elif isinstance(other, type(np.array(0))):
            dimOut2, dimOut1 = np.array([self.DimOutput]).reshape(-1), np.array(other.shape).reshape(-1)
            result.Expression = lambda u : mult(t(f(u)), other)
            result.Derivative = lambda u : mult(t(dfdu(u)), other)
            if result.FlagParMult and other.FlagParMult : result.Label = "<(" + str(other) +  ") , (" + self.Label + ")>"
            elif result.FlagParMult and not other.FlagParMult : result.Label = "<(" + str(other) + ") , " + self.Label + ">"
            elif not result.FlagParMult and other.FlagParMult : result.Label = "<" + str(other) + " , (" + self.Label + ")>"
            else :  result.Label =  "<" + str(other) + " , " + self.Label + ">"
            
        elif isinstance(other, VertexFunction):
            dimOut1, dimOut2 = np.array([self.DimOutput]).reshape(-1), np.array([other.DimOutput]).reshape(-1)
            g = deepcopy(other.Expression)
            dgdu =  deepcopy(other.Derivative)
            result.Expression = lambda u : mult(t(f(u)), g(u))
            result.Derivative = lambda u : mult(t(g(u)), dfdu(u)) + mult(t(f(u)), dgdu(u))
            if result.FlagParMult and other.FlagParMult : result.Label = "<(" + self.Label +  ") , (" +  other.Label + ")>"
            elif result.FlagParMult and not other.FlagParMult : result.Label = "<(" + self.Label + ") , " + other.Label + ">"
            elif not result.FlagParMult and other.FlagParMult : result.Label = "<" + self.Label + " , (" + other.Label + ")>"
            else :  result.Label =  "<" + self.Label + " , " + other.Label + ">"
        
        tst = mult(t(np.expand_dims(np.zeros(dimOut1), -1) ),np.expand_dims(np.zeros(dimOut2), -1))
        result.DimOutput = tst.shape[0:2]
        if tst.size == 1 : result.DimOutput = (1,)
        
        result.FlagParExp = True
        result.FlagParMult = False
        return result
    
    
    def __truediv__(self, other):
        result = deepcopy(self)
        if isinstance(other, (int, float)) :
            f = deepcopy(self.Expression)
            dfdu =  deepcopy(self.Derivative)
            result.Expression = lambda u : f(u) / other
            result.Derivative = lambda u : dfdu(u) / other
            
            if result.FlagParMult: result.Label =  "(" + self.Label + ") / " + str(other)
            else :  result.Label = self.Label+ " / " + str(other) 
        
        elif isinstance(other, VertexFunction) :
            assert self.DimInput == other.DimInput, "VertexFunctions should have the same InputDimension to perform division"
            # assert self.DimOutput == other.DimOutput, "Two VertexFunctions should have the same OutputDimension to be multiplied together" # may be ok because of broadcasting
            dimOut1, dimOut2 = np.array([self.DimOutput]).squeeze(), np.array([other.DimOutput]).squeeze()
            try: np.broadcast(np.zeros(dimOut1), np.zeros(dimOut2))
            except : raise Exception("VertexFunctions should be broadcastable to perform division") 
                
            f, g= deepcopy(self.Expression), deepcopy(other.Expression)
            dfdu, dgdu =  deepcopy(self.Derivative), deepcopy(other.Derivative)
            result.Expression = lambda u : f(u) / g(u)
            result.Derivative = lambda u : (dfdu(u)*g(u) - dgdu(u)*f(u))/(g(u)**2)
            result.DimOutput = np.broadcast(np.zeros(dimOut1), np.zeros(dimOut2)).shape

            if result.FlagParMult and other.FlagParMult : result.Label = "(" + self.Label + ") / (" + other.Label + ")"
            elif result.FlagParMult and not other.FlagParMult : result.Label = "(" + self.Label + ") / " + other.Label
            elif not result.FlagParMult and other.FlagParMult : result.Label =  self.Label + " / (" + other.Label + ")"
            else :  result.Label =  self.Label + " / " + other.Label
        result.FlagParExp = True
        result.FlagParMult = False
        return result
    
    def __rtruediv__(self, other):
        result = deepcopy(self)
        if isinstance(other, (int, float)) :
            dimOut2, dimOut1 = np.array([self.DimOutput]).squeeze(), np.array([other.DimOutput]).squeeze()
            try: np.broadcast(np.zeros(dimOut1), np.zeros(dimOut2))
            except : raise Exception("VertexFunctions should be broadcastable to perform division") 
            
            f = deepcopy(self.Expression)
            dfdu =  deepcopy(self.Derivative)
            result.Expression = lambda u : other / f(u)
            result.Derivative = lambda u : - other *  dfdu(u) / (f(u)**2)
            result.DimOutput = np.broadcast(np.zeros(dimOut1), np.zeros(dimOut2)).shape
            if result.FlagParMult: result.Label =  str(other) + "/ (" + self.Label + ")"
            else :  result.Label =  str(other) + " / " + self.Label
            result.FlagParExp = True
            result.FlagParMult = False
        return result
    
    def __pow__(self, other):
        result = deepcopy(self)
        if isinstance(other, (int, float)) :
            f = deepcopy(self.Expression)
            dfdu =  deepcopy(self.Derivative)
            result.Expression = lambda u : f(u) ** other
            result.Derivative = lambda u :dfdu(u) * other * f(u) ** (other-1)
            
            if result.FlagParExp: result.Label =  "(" + self.Label + ") ** " + str(other)
            else :  result.Label = self.Label+ " ** " + str(other) 
        
        elif isinstance(other, VertexFunction) :
            dimOut2, dimOut1 = np.array([self.DimOutput]).squeeze(), np.array([other.DimOutput]).squeeze()
            try: np.broadcast(np.zeros(dimOut1), np.zeros(dimOut2))
            except : raise Exception("VertexFunctions should be broadcastable to perform exponentiation") 
            
            f, g= deepcopy(self.Expression), deepcopy(other.Expression)
            dfdu, dgdu =  deepcopy(self.Derivative), deepcopy(other.Derivative)
            result.Expression = lambda u : f(u) ** g(u)
            result.Derivative = lambda u : dfdu(u)*g(u)*f(u)**(g(u)-1) + np.log(f(u))*dgdu(u)*f(u)**(g(u))

            result.DimOutput = np.broadcast(np.zeros(dimOut1), np.zeros(dimOut2)).shape
            if result.FlagParExp and other.FlagParExp : result.Label = "(" + self.Label + ") ** (" + other.Label + ")"
            elif result.FlagParExp and not other.FlagParExp : result.Label = "(" + self.Label + ") ** " + other.Label
            elif not result.FlagParExp and other.FlagParExp : result.Label =  self.Label + " ** (" + other.Label + ")"
            else :  result.Label =  self.Label + " ** " + other.Label
        result.FlagParExp = False
        result.FlagParMult = False
        return result
    
    def __rpow__(self, other):
        result = deepcopy(self)
        if isinstance(other, (int, float)) :
            dimOut1, dimOut2 = np.array([self.DimOutput]).squeeze(), np.array([other.DimOutput]).squeeze()
            try: np.broadcast(np.zeros(dimOut1), np.zeros(dimOut2))
            except : raise Exception("VertexFunctions should be broadcastable to perform exponentiation") 
            
            f = deepcopy(self.Expression)
            dfdu =  deepcopy(self.Derivative)
            result.Expression = lambda u : other ** f(u)
            result.Derivative = lambda u : other**f(u) * np.log(other) * dfdu(u)
            result.DimOutput = np.broadcast(np.zeros(dimOut1), np.zeros(dimOut2)).shape
            if result.FlagParExp: result.Label =  str(other) + " ** (" + self.Label + ")"
            else :  result.Label =  str(other) + " ** " + self.Label
            result.FlagParExp = False
            result.FlagParMult = False
        return result
 
    def plot(self, xmin = -1, xmax = 1, nPoints = 30, label3D = False):
        dimOut = tuple(np.hstack([self.DimOutput, 1]).reshape(-1))
        if self.DimInput == 1 :
            x = np.linspace(xmax, xmin, nPoints).reshape(1,1,-1)
            out = self.eval(x)
            for i in range(dimOut[0]):
                for j in range(dimOut[1]):
                    plt.plot(x.squeeze(), out[i,j,:].squeeze(), label = "$f_{"+str(i)+","+str(j)+"}$")
            plt.xlabel("u")
            plt.ylabel("f(u)")
            plt.legend(loc = 'best')
        elif self.DimInput == 2 :
            X = np.linspace(xmax, xmin, nPoints)
            Y =  np.linspace(xmax, xmin, nPoints)
            X, Y = np.meshgrid(X, Y)
            X, Y = X.reshape(1,1,-1), Y.reshape(1,1,-1)
            xy = np.vstack([X,Y])
            out =  self.eval(xy)
            plt.subplots(subplot_kw={"projection": "3d"})
            for i in range(dimOut[0]):
                for j in range(dimOut[1]):
                    plt.gca().plot_surface(xy[0,:,:].reshape(nPoints,nPoints), xy[1,:,:].reshape(nPoints,nPoints),
                                out[i,j,:].reshape(nPoints,nPoints), linewidth=0.1,
                                antialiased=True, label = "$f_{"+str(i)+","+str(j)+"}$")
            plt.gca().set_xlabel("$u_0$")
            plt.gca().set_ylabel("$u_1$")
            plt.gca().set_zlabel("$f(u)$")
            if label3D : plt.gca().legend(loc = 'best')     
        else : pass # TODO
        
        plt.grid()
    
    def __str__(self):
        return self.Label


# %%

# f1 = VertexFunction("f1", f = lambda u: np.array([[u], [2*u]]),
#                     dfdu = lambda u: np.array([[np.ones(u.shape)], [2*np.ones(u.shape)]]),
#                     dimOutput=(2))

# f2 = VertexFunction("f2", f = lambda u: np.array([[u], [2*u]]),
#                     dfdu = lambda u: np.array([[np.ones(u.shape)], [2*np.ones(u.shape)]]),
#                     dimOutput=(2))

# f = f1**f2
# f.plot()