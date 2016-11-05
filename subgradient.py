#Script for experiments with subgradient descent

#----------------------------------------------------------------------
#------------------------ Script Imports ------------------------------
#----------------------------------------------------------------------
#To make division easier
from __future__ import division
#For matrix and vector structures
import numpy as np
#Imports Pandas for data management
import pandas as pd
#for random numbers
import random
#For math operations
import math as math
#For system requirements
import sys
#For time measurement
import time

#For Graphing. The methods export the grapgics on plotly, the user only needs
# to enter his/her username and api-key
import plotly.plotly as py
import plotly.graph_objs as go
py.sign_in('minigonche', '8cjqqmkb4o') 


#----------------------------------------------------------------------
#------------------------ Global Variables ----------------------------
#----------------------------------------------------------------------

#Imports the data values stored in 'data/Datos.csv' for the corresponding
# x_i and y_i

data_x = np.matrix(pd.DataFrame.from_csv('data/Datos.csv', index_col = None))
dim_data = data_x.shape[1]
data_y = data_x[:,data_x.shape[1] - 1]
data_x = data_x[:,0:(data_x.shape[1] - 1)]

data_x = np.matrix([[1,2],[3,4],[5,6]])
print(data_x)
data_y = np.matrix([[1],[0],[1]])
dim_data = 3

lambda_value = 1

n = 500
m = 200
c = np.random.rand(1,n)
#Puts each a_j as a column of the following matrix
A = np.random.rand(n,m)
#Global constant alpha
global_alpha = 0.001
#GLobal epsilon for treshold
global_eps = 0.001
#global difference measure for gradient
global_dif = 0.000001
#Measure how many iterations to print pogress
print_counter = 20


#----------------------------------------------------------------------
#------------------------ Main Methods --------------------------------
#----------------------------------------------------------------------

#NOTE: Vectors are assumed as matrix of dimension 1 x n
#Runs the subgradient descent with the given parameters
#Serves as a unified method
def run_subgradient_descent(dim, fun, subgradient, alpha, eps, initial = None):
    """
        Parameters
        ----------
        dim : int
            The dimension of the vector that the function and subgradient receive.
            The domain's dimension of the function we wish to minimize
        fun : function(numpy.vector)
            The function we wish to minimize
        subgradient : functon(numpy.vector)
            A functon that receives a numpy.vecotr (real vector: x_k) and 
            returns the gradient (as a numpy.vector) of the given function 
            evaluated at the real number received as parameter
        alpha : function(numpy.vector, numpy.vector, float)
            A functon that receives two numpy.vecotors (real vectors: 
            x_k and g_k ) and the previuos alph to return the next alpha step
        eps : float
            The epsylon that serves as a stopping criteria for the algorithm
        Initial : np:vector
            The initial vector. If None is received, then the procedure strarts
            at zero.
    """
    #Starts the timer
    start_time = time.time()

    #Initial values
    #The first alpha and B matrix are initialized at None
    
    x = initial
    if x is None:
        x = np.zeros((1,dim))
        
    x_last = np.zeros((1,dim))
    subgrad_last = np.zeros((1,dim))
    a = None
    g = None
    
    #Minimum Value of the function
    min_x = x
    min_value = fun(x)
    
    #Treshold
    treshold = False
    
    #printing variables
    count = 1
    global_count = 0
    
    #Graphing variables
    x_variables = [x]
    function_values = [fun(x)]
    
    
    #Becomes true when |f(x_n+1) - f(x_n)| < eps
    while(not treshold):

        #Calculates the necesarry advancing parameters
        x_actual = x
        g = subgradient(x_actual)


        a = alpha(x_actual, g, a)
        x = x_actual - a*(g/np.linalg.norm(g))
        x_last = x_actual
        
        #Checks the the treshold
        treshold = global_count > 100000
        
        if count == print_counter:
            print(x)
            print(temp_value)
            print(min_value)
            count = 0
        
        count = count + 1
        global_count = global_count +1
        subgrad_last = g
        
        #Saves the current x
        x_variables.append(x)
        #Calcultaes the value
        temp_value = fun(x)
        #Appends the calculated value
        function_values.append(temp_value)
        #Refreshes the global minimum
        if(temp_value < min_value):
            min_x = x
            min_value = temp_value
        
    
    return [min_x, min_value, x_variables, function_values, global_count, time.time() - start_time]
#end of run_subgradient_descent

#The proximal gradient method
def run_proximal_gradient_descent(dim, fun, prox_fun, gradient, alpha, initial = None ):
    """
        Parameters
        ----------
        dim : int
            The dimension of the vector that the gradient receive.
            The domain's dimension of the function we wish to minimize
        fun : function(numpy.vector)
            A function of the form g(x) + h(x) where both are continuos but
            only g is defirentiable
        prox_fun : function(numpy.vector)
            The proximity function that will be applied to h(x)
        gradient : functon(numpy.vector)
            The gradient function of g(x)
        alpha : function(numpy.vector, numpy.vector, float)
            A functon that receives two numpy.vecotors (real vectors: 
            x_k and g_k ) and the previuos alph to return the next alpha step
        Initial : np:vector
            The initial vector. If None is received, then the procedure strarts
            at zero.    
    """
    #Starts the timer
    start_time = time.time()

    #Initial values
    #The first alpha and B matrix are initialized at None
    
    x = initial
    if x is None:
        x = np.zeros((1,dim))
        
    x_last = np.zeros((1,dim))
    grad_last = np.zeros((1,dim))
    a = None
    g = None
    
    #Treshold
    treshold = False
    
    #printing variables
    count = 1
    global_count = 0
    
    #Graphing variables
    x_variables = [x]
    function_values = [fun(x)]
    
    #Becomes true when the iterations are exceeded
    while(not treshold):

        #Calculates the necesarry advancing parameters
        x_actual = x
        g = gradient(x_actual)

        a = alpha(x_actual, g, a)
        
        G = (x_actual - prox_fun(x_actual - a*g))/a
        
        x = x_actual - a*G
        x_last = x_actual
        
        #Checks the the treshold
        treshold = global_count > 10000

        grad_last = g
        
        #Saves the current x
        x_variables.append(x)
        #Calculates value
        temp_value = fun(x)
        #Appends the calculated value
        function_values.append(temp_value)
        
        if count == print_counter:
            print(temp_value)
            count = 0
            
        count = count + 1
        global_count = global_count +1

    final_x = x
    final_value = temp_value
    
    return [final_x, final_value, x_variables, function_values, global_count, time.time() - start_time]
    




#----------------------------------------------------------------------
#------------------------ Experiment Start ----------------------------
#----------------------------------------------------------------------


#Declares the subgradient of the absolute value
def subgradient_abs(x_single):
    if x_single > 0:
        return 1
    if x_single < 0:
        return -1

    return random.uniform(-1, 1)
    

    
#Declares the subgradient of the norm_1
def subgradient_norm1(x_vec):
    
    return np.array(map(subgradient_abs, x_vec.T)).T
    
    
#Proximity function
#View hand-in for construction
def prox(t,x):
    
    def coordinate_prox(x_i):
        if x_i > t:
            return x_i - t
        if x_i < - t:
            return x_i + t
            
        return 0    
    
    return np.array(map(coordinate_prox, x.T)).T    
    
    
    
        
#CENTRAL FUNCTION
#Declares the log(1 + exp(x)) so it can handel large numbers
def log_exp(x):
    if x > 705:
        return x
    return  np.log(1 + np.exp(x))   
#end log_exp

#Declares the function exp(x)/ (1 + exp(x)) so it can handle large numbers
def exp_over_exp(x):
    if x > 36:
        return 1
    return np.exp(x)/(1 + np.exp(x))
#end exp_over_exp    
  

#Declares the global function, its gradient and its Hessian
def main_function(beta):
    
    #Column vector
    x_beta = np.dot(data_x, beta.T)
    
    first_term = np.dot(x_beta.T, (-1)*data_y)[0,0]
    
    second_term = sum(map(lambda k: log_exp(k[0,0]) , x_beta ))
    
    third_term = lambda_value*np.linalg.norm(beta.T, 1)

    return(first_term + second_term + third_term)
#end of main_function   

def main_subgradient(beta):
    
    x_beta = np.dot(data_x, beta.T)
    
    #first constructs the vector y_i + exp()/(1 + exp)
    #Constructs the gradient
    temp_vec = (-1)*data_y.T + map(lambda k: exp_over_exp(k[0,0]), x_beta)
    first_term = np.dot(temp_vec, data_x)
    
    
    #Constructs the subgradient
    second_term = lambda_value*subgradient_norm1(beta)   


    return(first_term + second_term)
#end of main_gradient

def alpha_fun(x,g,a):
    if a == None:
        return 1
    
    return 1/(1/a + 0.1)

result = run_subgradient_descent(dim_data -1, main_function, main_subgradient, alpha_fun, 0.00001, initial = None)

print(result[1])



    