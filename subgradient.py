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

#data_x = np.matrix(pd.DataFrame.from_csv('data/Datos.csv', index_col = None))
#dim_data = data_x.shape[1]
#data_y = data_x[:,data_x.shape[1] - 1]
#data_x = data_x[:,0:(data_x.shape[1] - 1)]
#n = data_x.shape[0]

data_x = np.matrix([[1,2],[3,4],[5,6]])
data_y = np.matrix([[1],[0],[1]])
dim_data = 3
n = 3

#lambda value
lambda_value = 1

#Global constant alpha
global_alpha = 0.001
#GLobal epsilon for treshold
global_eps = 0.001
#Measure how many iterations to print pogress
print_counter = 20
#maximimum iteration
max_ite = 1000
#global difference measure for gradient
global_dif = 0.000001


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
    
    
    #Becomes true when the iterations are exceeded
    while(not treshold):

        #Calculates the necesarry advancing parameters
        x_actual = x
        g = subgradient(x_actual)

        a = alpha(x_actual, g, a)
        x = x_actual - a*(g/np.linalg.norm(g.T))
        x_last = x_actual
        
        #Checks the the treshold
        treshold = global_count > max_ite
        
        if count == print_counter:
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
def run_proximal_gradient_descent(dim, fun, prox_fun, gradient, alpha, eps, initial = None ):
    """
        Parameters
        ----------
        dim : int
            The dimension of the vector that the gradient receive.
            The domain's dimension of the function we wish to minimize
        fun : function(numpy.vector)
            A function of the form h(x) = f(x) + g(x) where both are continuos but
            only f is defirentiable
        prox_fun : function(numpy.vector)
            The proximity function that will be applied to h(x)
        gradient : functon(numpy.vector)
            The gradient function of f(x)
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
    
    #Becomes true when the iterations are exceeded or when |G| < eps
    while(not treshold):

        #Calculates the necesarry advancing parameters
        x_actual = x
        g = gradient(x_actual)

        a = alpha(x_actual, g, a)
        
        G = (x_actual - prox_fun(a,x_actual - a*g))/a
        
        x = x_actual - a*G
        x_last = x_actual
        
        #Checks the the treshold
        treshold = global_count > max_ite or np.linalg.norm(G)< eps

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
    

#Runs the ADMM method
#Minimizes a function of the form sum(f)+ g
def run_ADMM(dim, f_array, array_gradient_f, g, subgradient_g, alpha, eps, initial = None):
    """
        Parameters
        ----------
        dim : int
            The dimension of the vector that the function and subgradient receive.
            The domain's dimension of the function we wish to minimize
        f_array : array of functions of the form: function(numpy.vector)
            The array of the differential functions that will be added
        array_gradient_f : array of funtions of the form functon(numpy.vector)
            The array of the gradient functions of each f_i(x)
        g : function(numpy.vector)
            The non-diferential portion fo the function we wish to minimize
        subgradient_f : functon(numpy.vector)
            The subgradient  function of g(x)                
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

    #declares the nunmber of functions in the array
    n = len(f_array)

    #Initial values
    #The first alpha and beta matrix are initialized at None
    mu = initial
    if mu is None:
        mu = np.zeros((1,dim))
    
    a = None    
    beta = np.zeros((1,dim))
    nu = np.zeros((1,dim))    
    mu_array = n*[np.zeros((1,dim))]
    nu_array = n*[np.zeros((1,dim))]
    
    #Treshold
    treshold = False
    
    #printing variables
    count = 1
    global_count = 0
    
    #Graphing variables
    x_variables = [mu]
    function_values = [sum(map(lambda f: f(mu),f_array)) + g(mu)]

    #Declares the method that return each indivudual functon to be minimized
    def get_f(i):
        def f_i(nu):
            first_term = f_array[i](nu)
            second_term = (a/2)*(np.linalg.norm(nu - beta + mu_array[i] )**2)
            return first_term + second_term

        return f_i

    #Fills the array with the corresponding functions    
    nu_functions = []    
    nu_functions = map(get_f, range(n))    
    
    #Declares teh method that return each individual gradient    
    def get_gradient_f(i):
        def gradient_i(nu):
            first_term = array_gradient_f[i](nu)
            second_term = a*np.linalg.norm(nu - beta + mu_array[i] )
            return first_term + second_term

        return gradient_i

    #fills the array with the corresponding gradients
    nu_gradients = []
    nu_gradients = map(get_gradient_f, range(n))     

    #declares the beta function that will also be needed to minimized
    def beta_fun(x):
        first_term = g(x)
        second_term = (n*a/2)*(np.linalg.norm(x - nu - mu ))**2
        return first_term + second_term

    #declares teh beta function that will also be needed to minimized
    def beta_subgrad(x):
        first_term = G_subgradient(x)
        second_term = (n*a)*(np.linalg.norm(x - nu - mu ))
        return first_term + second_term


    #Becomes true when the iterations are exceeded
    while(not treshold):

    	a = alpha(mu,beta,a)
        #Calculates every nu
        for i in range(n):
        	nu_array[i] = run_gradient_descent(dim, 
                                               nu_functions[i], 
                                               nu_gradients[i], 
                                               alpha = lambda x, p: global_alpha,
                                               B_matrix = lambda B, x, x_prev: np.identity(dim), 
                                               eps = global_eps, 
                                               inverse = True, 
                                               initial = None)[0]
        #Finds \hat nu_{k+1}       
        nu = sum(nu_array)/n
        
        #Finds beta_{k+1}
        beta = run_subgradient_descent(dim, beta_fun, beta_subgrad, alpha_fun_decs, 0.00001, initial = None)[0]	

        mu_array = map(lambda i: mu_array[i] + nu_array[i] - beta, range(n))

        #Finds \hat mu_{k+1}
        mu = sum(mu_array)/n       
        
        #Checks the the treshold
        treshold = global_count > max_ite
        
        if count == print_counter:
            print(temp_value)
            print(min_value)
            count = 0
        
        count = count + 1
        global_count = global_count +1
        subgrad_last = g
        
        #Saves the current x
        x_variables.append(mu)
        #Calcultaes the value
        temp_value = sum(map(lambda f: f(mu),f_array)) + g(mu)
        #Appends the calculated value
        function_values.append(temp_value)

    x_final = mu
    value_final = sum(map(lambda f: f(mu),f_array)) + g(mu)    

    
    return [x_final, value_final, x_variables, function_values, global_count, time.time() - start_time]
#end of run_ADMM


#Runs the gradient descent with the given parameters
#Serves as a unified method
def run_gradient_descent(dim, fun, gradient, alpha, B_matrix, eps, inverse = True, initial = None):
    """
        Parameters
        ----------
        dim : int
            The dimension of the vector that the function and gradient receive.
            The domain's dimension of the function we wish to minimize
        fun : function(numpy.vector)
            The function we wish to minimize
        gradient : functon(numpy.vector)
            A functon that receives a numpy.vecotr (real vector: x_k) and 
            returns the gradient (as a numpy.vector) of the given function 
            evaluated at the real number received as parameter
        alpha : function(numpy.vector, numpy.vector)
            A functon that receives two numpy.vecotors (real vectors: 
            x_k and p_k ) and returns the next alpha step
        B_matrix : function(np.matrix, numpy.vector)
            A function that receives a numpy.matrix (the previous matrix) and 
            numpy.vecotr (real vector) and returns the next multiplication
            np.matrix 
        eps : float
            The epsylon that serves as a stopping criteria for the algorithm
        solve : boolean
            Indicates if the B_matrix method gives the B or the B^-1 matrix
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
    B = None
    a = None
    p = None
    
    #Treshold
    treshold = False
    
    #printing variables
    count = 1
    global_count = 0
    
    #Graphing variables
    x_variables = []
    function_values = []
    
    
    #Becomes true when |f(x_n+1) - f(x_n)| < eps
    while(not treshold):
        #Saves the Value
        x_variables.append(x)
        function_values.append(fun(x))
        
        #Calculates the necesarry advancing parameters
        x_actual = x
        
        B = B_matrix(B, x_actual,x_last)
        grad = gradient(x_actual)

        #Calcultaes the next value
        if inverse:
            p = (-1)*B.dot(grad.T).T
        else:
            p = (-1)*np.linalg.solve(B, grad.T).T
            
        
        #raw_input('espere')    
        
        a = alpha(x_actual, p)
        x = x_actual + a*p
        x_last = x_actual
        
        #Checks the the treshold
        treshold = np.linalg.norm(grad) < eps  or np.linalg.norm(grad - grad_last) < global_dif
        
        if count == print_counter:
            print(np.linalg.norm(grad))
            count = 0
        
        count = count + 1
        global_count = global_count +1
        grad_last = grad
        
    
    x_final = x
    value_final = fun(x)
    
    
    return [x_final, value_final, x_variables, function_values, global_count, time.time() - start_time]
#end of run_gradient_descent



#----------------------------------------------------------------------
#------------------------ Support Functions ---------------------------
#----------------------------------------------------------------------



#Declares the subgradient of the absolute value
def subgradient_abs(x_single):
    if x_single > 0:
        return 1
    if x_single < 0:
        return -1

    return 1/2    
    #return random.uniform(-1, 1)
    
           
#Proximity function
#View hand-in for construction
def prox(t,x):
    
    def coordinate_prox(x_i):
        if x_i > t:
            return x_i - t
        if x_i < - t:
            return x_i + t
            
        return 0    
    
    return np.array(map(lambda k: coordinate_prox(k[0,0]), x.T)).T    
#end of prox

#declares the gloal constant alpha function
def alpha_fun_cons(x,g,a):
    return global_alpha
#end of alpha_fun_dec


#declares the gloal decreasing alpha function
def alpha_fun_decs(x,g,a):
    if a == None:
        return 1
    
    return 1/(1/a + 1)
#end of alpha_fun_dec

def alpha_fun_back(x,g,a):
    #For the first iteration
    if(g is None):
        return global_alpha

    rho = 4/5
    c = 4/5        
    a = 1

    while(H(x + a*g) > H(x) + c*a*np.dot(H_subgradient(x),g.T) ):
        a = rho*a
    
    return a
# end of alpha_backtracking

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

#----------------------------------------------------------------------
#-------------------------- Main Functions ----------------------------
#----------------------------------------------------------------------    

def F(beta):
	#Column vector
    x_beta = np.dot(data_x, beta.T)
    
    first_term = np.dot(x_beta.T, (-1)*data_y)[0,0]
    
    second_term = sum(map(lambda k: log_exp(k[0,0]) , x_beta ))
        
    return(first_term + second_term)
#end of F    

def F_gradient(beta):
    
    x_beta = np.dot(data_x, beta.T)

    #first constructs the vector y_i + exp()/(1 + exp)
    #Constructs the gradient
    temp_vec = (-1)*data_y.T + map(lambda k: exp_over_exp(k[0,0]), x_beta)
    first_term = np.dot(temp_vec, data_x)
        
    return(first_term)
#end of F_gradient

#Since f is differentiable, its subgradient is the gradient
def F_subgradient(beta):
	return F_gradient(beta)
#end of F_subgrdient	

#declares the function g(x) = lamnda*|beta|
def G(beta):
	return lambda_value*np.linalg.norm(beta.T, 1)
#end of G

#Declares the subgradient of the fuction G
def G_subgradient(x_vec):    
    return lambda_value*np.array(map(subgradient_abs, x_vec.T)).T
#end of G_subgradient

#Declares the global function h, its subgradient 
def H(beta):
	return F(beta) + G(beta)
     
#end of main_function   

#Declares the subgradient fo h(x)
def H_subgradient(beta):            
    return(F_gradient(beta) + G_subgradient(beta))
#end of H_subgradient

#Declares a stochastic subgradient for the dimentions of h(x)
def H_subgradient_stoc(beta):
       
    return np.random(())
#end of H_subgradient_stoc



#result = run_subgradient_descent(dim_data -1, H, H_subgradient, alpha_fun_decs, 0.00001, initial = None)
#result = run_proximal_gradient_descent(dim_data-1, H, prox, F_gradient, alpha_fun_decs, 0.00001, initial = None )



def construct_f(i):
    def f_i(beta):
    	x_beta = np.dot(data_x[i,:],beta.T)[0,0]
    	return (-1)*data_y[i,0]*x_beta + log_exp(x_beta)

    return f_i	

    array.append(f_i)
def construct_grad_f(i):
    def subgrad_f_i(beta):
    	x_beta = np.dot(data_x[i,:],beta.T)[0,0]
    	return ((-1)*data_y[i,0] + exp_over_exp(x_beta))*data_x[i,:]

    return subgrad_f_i	

array_f = map(construct_f, range(n))
grad_array_f = map(construct_grad_f, range(n))

result = run_ADMM(dim_data - 1, array_f, grad_array_f, G, G_subgradient, alpha_fun_cons, global_eps, initial = None)

print(result[1])
print(result[4])

'''
x = np.zeros((1,3))
x[0,0] = 1
x[0,1] = 1
x[0,2] = 1

print F_gradient(x)

'''


    