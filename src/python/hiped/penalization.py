import numpy as np
import matplotlib.pyplot as plt
   
class Penalization:
    """
    A class used to represent a penalization function.
    
    Feel free to add your own custom penalization functions to the library in 
    the __init__ method.

    Attributes
    ----------
    
    CoeffPenal : float
        Penalization coefficient
        
    Type : str
        Label of the penalization
        
    Reference : str
        Litterature related to the penalization
        
    Expression : function
        Expression of the penalization function
    
    Derivative : function
        Expression of the derivative of the penalization function
        
    Methods
    -------
    
    eval(x)
        Compute the penalization evaluated on x
        
    evald(x)
         Compute the penalization derivative evaluated on x 
        
    plot(derivative = False)
        Plot the penalization or its derivative on [0,1] interval.
            
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
    
    def __init__(self, penalType = "simp", coeffPenal = 1, reverse = False, reference = "",
                 derivative1 = 1, derivative2 = 1):
        
        self.CoeffPenal = coeffPenal
        self.Type = penalType
        if reverse: 
            self.Type = "1 - " + self.Type + "(1-x)"
            
        
        if penalType.lower() in ["simp", "power", "powerlaw", "power law", "pow", "power_law"]:
            
            self.Reference = "Mlejnek (1992), Bendsøe & Sigmund (1999), Sigmund (2003)"
            self.Expression= lambda x: np.real(x**self.CoeffPenal)
            self.Derivative= lambda x: np.real(self.CoeffPenal * x**(self.CoeffPenal-1))
            if reverse:
                self.Expression= lambda x: 1 - np.real((1-x)**self.CoeffPenal)
                self.Derivative= lambda x: np.real(self.CoeffPenal * (1-x)**(self.CoeffPenal-1))
        
        if penalType.lower() in ["ramp", "r", "rat", "rational"]:
            
            self.Reference = "Stolpe and Svanberg (2001), Hansen (2005)"
            self.Expression= lambda x: np.real(x / (self.CoeffPenal*(1-x) + 1))
            self.Derivative= lambda x: np.real((self.CoeffPenal + 1)/(self.CoeffPenal*(1-x) + 1)**2)
            if reverse:
                self.Expression= lambda x: 1-np.real( (1-x) / (self.CoeffPenal*x + 1))
                self.Derivative= lambda x: - np.real((x*(x-1)) / (self.CoeffPenal*x + 1 )**2)
        
        if penalType.lower() in ["lukas", "lukáš", "atan", "arctan", "heaviside"]:
            
            self.Reference = "Lukáš (2006) - An Integration of Optimal Topology and Shape Design for Magnetostatics"
            self.Expression= lambda x: np.real((1 + np.arctan(self.CoeffPenal*(2*x-1))/np.arctan(self.CoeffPenal))/2)
            self.Derivative= lambda x: np.real(self.CoeffPenal/(np.arctan(self.CoeffPenal) * (self.CoeffPenal**2 * (2*x - 1)**2 + 1)))
            # no change for this penalization type when reversed

        if penalType.lower() in ["zhou", "rational2","rat2", "zhu"]:    
            self.Expression= lambda x: np.real(x/self.CoeffPenal + (self.CoeffPenal-1)/self.CoeffPenal*x**3)
            self.Derivative = lambda x: np.real(1/self.CoeffPenal + (3*x**2*(self.CoeffPenal-1))/self.CoeffPenal)
            self.reference = "don't remember";
            if reverse:
                self.Expression= lambda x: 1 - np.real((1-x)/self.CoeffPenal + (self.CoeffPenal-1)/self.CoeffPenal*(1-x)**3)
                self.Derivative= lambda x: np.real(1/self.CoeffPenal + 3*(self.CoeffPenal - 1)*(x - 1)**2 / self.CoeffPenal)
                
        # To continue with other penalizations
    
    def eval(self, x):
        return self.Expression(x)
    
    def evald(self, x):
        return self.Derivative(x)
    
    
    def plot(self, derivative = False):
        x = np.arange(0,1.01,0.01)
        if not derivative:
            plt.plot(x,self.eval(x), label = f" p = {self.CoeffPenal}")
            plt.ylabel("$P(\omega)$")
        else:
            plt.plot(x,self.deval(x), label = f" p = {self.CoeffPenal}")
            plt.ylabel("$dP/d\omega$")
        plt.grid()
        plt.legend()
        plt.xlabel("$\omega$")
        plt.axis("equal")
        

