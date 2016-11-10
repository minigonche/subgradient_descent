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
#For parallelism
from multiprocessing import Pool


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
n = data_x.shape[0]

max_c = np.amax(data_x,0)
for i in range(dim_data-1):
    data_x[:,i] = (1/max_c[0,i])*data_x[:,i]



'''

data_x = np.matrix([[1,2],[3,4],[5,6]])
data_y = np.matrix([[1],[0],[1]])
dim_data = 3
n = 3

'''

#lambda value
lambda_value = 1

#Global constant alpha
global_alpha = 0.001
#GLobal epsilon for treshold
global_eps = 0.001
#Measure how many iterations to print pogress
print_counter = 3
#maximimum iteration
max_ite = 30000
#global difference measure for gradient
global_dif = 0.000001
#Global alpha step
alpha_step = 1
#Stochastic percentage to calculate th number of rows
#to be included in the stochastic method
stochastic_percentage = 0.6
#Number of values that need to be smaller than eps to converge
series = 25
#Max_unaltered
max_unaltered = 300


#----------------------------------------------------------------------
#------------------------ Main Methods --------------------------------
#----------------------------------------------------------------------

#NOTE: Vectors are assumed as matrix of dimension 1 x n
#Runs the subgradient descent with the given parameters
#Serves as a unified method
def run_subgradient_descent(dim, fun, subgradient, alpha, eps, initial = None, print_progress = True):
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
        print_progress : boolean
            Boolean indicating if the procedure should print its progress
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
    min_value = np.inf  
    
    #Treshold
    treshold = False
    
    #printing variables
    count = 1
    global_count = 0
    #The number of values than are smaller than eps
    eps_count = 0
    
    #Graphing variables
    x_variables = [min_x]
    function_values = [min_value]

    unaletered = 0
    
    
    #Becomes true when the iterations are exceeded
    while(not treshold):

        #Calculates the necesarry advancing parameters
        x_actual = x
        g = subgradient(x_actual)

        p = (-1)*g/np.linalg.norm(g.T)

        a = alpha(x_actual, p, a)
        x = x_actual + a*p
        x_last = x_actual
        
        if print_progress and count == print_counter:
            print(temp_value)
            print(min_value)
            print eps_count
            print unaletered
            count = 0
        
        count = count + 1
        global_count = global_count +1
        subgrad_last = g
        

        #Calcultaes the value
        temp_value = fun(x)        
        


        #Refreshes the global minimum
        if(temp_value < min_value):
            unaletered = 0
            if np.linalg.norm(temp_value - min_value) < eps:
                eps_count = eps_count +1
            else:
                eps_count = 0

            min_x = x
            min_value = temp_value
        else:
            unaletered = unaletered +1     

        #Checks the the treshold
        treshold = global_count > max_ite or eps_count > series  or unaletered > max_unaltered  
        #Saves the current minimum value
        x_variables.append(min_x)                
        function_values.append(min_value)                
        
    
    return [min_x, min_value, x_variables, function_values, global_count, time.time() - start_time]
#end of run_subgradient_descent

#The proximal gradient method that will be used with backtracking
def run_proximal(dim, fun, prox_fun, gradient, alpha, eps, initial = None , print_progress = True):
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
        print_progress : boolean
            Boolean indicating if the procedure should print its progress    
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
    a = global_alpha
    g = None
    
    #Treshold
    treshold = False
    
    #printing variables
    count = 1
    global_count = 0
    #The number of values than are smaller than eps
    eps_count = 0
    
    #Graphing variables
    x_variables = [x]
    function_values = [fun(x)]
    temp_value = fun(x)
    
    #Becomes true when the iterations are exceeded or when |G| < eps
    while(not treshold):

        #Calculates the necesarry advancing parameters
        x_actual = x
        g = gradient(x_actual)

        a = alpha(x_actual, g, a)

        G = (-1)*(x_actual - prox_fun(a, x_actual - a*g))/a

        
        
        x = x_actual + a*G
        x_last = x_actual
        
        if np.linalg.norm(temp_value - fun(x)) < eps:
            eps_count = eps_count +1
        else:
            eps_count = 0            
        
        #Checks the the treshold
        treshold = global_count > max_ite or np.linalg.norm(G)< eps or eps_count > series

        grad_last = g
        
        #Saves the current x
        x_variables.append(x)
        #Calculates value
        temp_value = fun(x)
        #Appends the calculated value
        function_values.append(temp_value)
        
        if print_progress and count == print_counter:
            print(temp_value)
            print eps_count
            count = 0
            
        count = count + 1
        global_count = global_count + 1

    final_x = x
    final_value = temp_value
    
    return [final_x, final_value, x_variables, function_values, global_count, time.time() - start_time]
#end of run_proximal

#The proximal gradient method that will be used with backtracking
def run_proximal_accelerated(dim, fun, prox_fun, gradient, alpha, eps, initial = None , print_progress = True):
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
        print_progress : boolean
            Boolean indicating if the procedure should print its progress    
    """
    #Starts the timer
    start_time = time.time()

    #Initial values
    #The first alpha and B matrix are initialized at None
    
    x = initial
    if x is None:
        x = np.zeros((1,dim))
        
    x_last = x
    v = x

    a = global_alpha
    g = None
    
    #Treshold
    treshold = False

    #Minimum Value of the function
    min_x = x
    min_value = fun(x)
    
    
    #printing variables
    count = 1
    global_count = 1
    #The number of values than are smaller than eps
    eps_count = 0
    
    #Graphing variables
    x_variables = [x]
    function_values = [fun(x)]
    temp_value = fun(x)
    
    #Becomes true when the iterations are exceeded or when |G| < eps
    while(not treshold):

                
        g = gradient(v)

        a = alpha(x, g, v)     
        
        x = prox_fun(a,v - a*g)        

        v = x + ((global_count - 1)/(global_count + 2))*(x - x_last)

        if np.linalg.norm(temp_value - fun(x)) < eps:
            eps_count = eps_count +1
        else:
            eps_count = 0  

        #Checks the the treshold
        treshold = global_count > max_ite or eps_count > series


        
        #Saves the current x
        x_variables.append(x)
        #Calculates value
        temp_value = fun(x)
        #Appends the calculated value
        function_values.append(temp_value)
        
        if print_progress and count == print_counter:
            print(temp_value)
            print(min_value)
            print eps_count
            count = 0
            
        count = count + 1
        global_count = global_count +1

        #Refreshes the global minimum
        if(temp_value < min_value):
            min_x = x
            min_value = temp_value

        x_last = x      

    return [min_x, min_value, x_variables, function_values, global_count, time.time() - start_time]
#end of run_proximal_accelerated
    

#Runs the ADMM method
#Minimizes a function of the form sum(f)+ g
def run_ADMM(dim, f_array, array_gradient_f, g, subgradient_g, alpha, eps, initial = None, print_progress = True, parallel = False):
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
        print_progress : boolean
            Boolean indicating if the procedure should print its progress    
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

    #The last minimum value found
    last_min = np.inf
    #the current minimum
    min_value = np.inf
    
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
    nu_functions = map(get_f, range(n))    
    
    #Declares teh method that return each individual gradient    
    def get_gradient_f(i):
        def gradient_i(nu):
            first_term = array_gradient_f[i](nu)
            second_term = a*(nu - beta + mu_array[i] )            
            return first_term + second_term

        return gradient_i

    #fills the array with the corresponding gradients    
    nu_gradients = map(get_gradient_f, range(n))

    #Constructs the arguments for the parallel case
    parallel_args = map(lambda i: [get_f(i),get_gradient_f(i)], range(n))    

    #declares the beta function that will also be needed to minimized
    def beta_fun(x):
        first_term = g(x)
        second_term = (n*a/2)*(np.linalg.norm(x - nu - mu ))**2
        return first_term + second_term

    #declares teh beta function that will also be needed to minimized
    def beta_subgrad(x):
        first_term = G_subgradient(x)
        second_term = (n*a)*(x - nu - mu )
        return first_term + second_term

    #Becomes true when the iterations are exceeded
    while(not treshold):

        last_nu = nu
    	a = alpha(mu,beta,a)
        #Calculates every nu
        
        if parallel:
            #Parallel            
            pool = Pool(n)
            nu_array = pool.map(parallel_min, map(lambda i:[dim,i,a,beta,mu_array] ,range(n)))
            pool.close()
            

        else:        
            #Linear
            for i in range(n):
            	nu_array[i] = parallel_min([dim,i,a,beta,mu_array])


        #Finds \hat nu_{k+1}       
        nu = sum(nu_array)/n
        
        #Finds beta_{k+1}
        beta = prox(1/(n*a),nu + mu)

        mu_array = map(lambda i: mu_array[i] + nu_array[i] - beta, range(n))

        #Finds \hat mu_{k+1}
        mu = sum(mu_array)/n
        
        if print_progress and count == print_counter:
            print(min_value)
            count = 0
        
        count = count + 1
        global_count = global_count +1
        subgrad_last = g

        
        x = nu
        last_min = min_value
        #Saves the current x
        x_variables.append(x)
        #Calcultaes the value
        min_value = sum(map(lambda f: f(x),f_array)) + g(x)
        #Appends the calculated value
        function_values.append(min_value)

        #Checks the the treshold
        treshold = global_count > max_ite or math.fabs(min_value - last_min) < 0.001

    x_final = x
    value_final = sum(map(lambda f: f(x),f_array)) + g(x)    

    
    return [x_final, value_final, x_variables, function_values, global_count, time.time() - start_time]
#end of run_ADMM


#Runs the gradient descent with the given parameters
#Serves as a unified method
def run_gradient_descent(dim, fun, gradient, alpha, B_matrix, eps, inverse = True, initial = None, print_progress = True):
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
        print_progress : boolean
            Boolean indicating if the procedure should print its progress
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
        
        a = alpha(x_actual, p, a)
        x = x_actual + a*p
        x_last = x_actual
        
        #Checks the the treshold
        treshold = np.linalg.norm(grad) < eps  or np.linalg.norm(grad - grad_last) < global_dif
        
        
        if  print_progress and count == print_counter:
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

#Declares the global method for the parallel procedure.
#This method should only acces global variables and should
#not invoke any functions

def parallel_min(args):

    #Variables for the iteration
    dim = args[0]
    i = args[1]
    a_k = args[2]
    beta = args[3]    
    mu_array = args[4]


    #Variables for the gradient descent
    B = np.identity(dim)
    eps = 0.01
    a_grad = 0.0001




    #Excecutes gradient descent
    eta = np.zeros((1,dim))        
    alpha = a_grad    
    
    #Treshold
    treshold = False
        
    
    #Becomes true when |\nabla f(x)| < eps
    while(not treshold):
        
        #calculates the gradient
        x_eta = np.dot(data_x[i,:],eta.T)[0,0]

        first_term =  ((-1)*data_y[i,0] + exp_over_exp(x_eta))*data_x[i,:]        
        second_term = a_k*(eta - beta + mu_array[i] )            
        
        grad = first_term + second_term

        #calculates alpha using backtracking        
        rho = 4/5
        c = 3/5        
        alpha = 1

        back_treshold = True

        while(back_treshold):
            #regreshes alpha
            alpha = rho*alpha
            v_left = eta + alpha*(-1)*grad
            x_left = np.dot(data_x[i,:],v_left.T)[0,0]    
            left = (-1)*data_y[i,0]*x_left + log_exp(x_left) + (a_k/2)*(np.linalg.norm(v_left - beta + mu_array[i] )**2)

            right =   (-1)*data_y[i,0]*x_eta + log_exp(x_eta) + (a_k/2)*(np.linalg.norm(eta - beta + mu_array[i] )**2)  

            back_treshold = left > right + c*alpha*np.dot(grad,(-1)*grad.T)



        eta = eta + alpha*(-1)*grad
        
        #Checks the the treshold
        treshold = np.linalg.norm(grad) < eps 
                
    return eta    
    
    



#Declares the subgradient of the absolute value
def subgradient_abs(x_single):
    if x_single > 0:
        return 1
    if x_single < 0:
        return -1

    return 1/2    
#end of subgradient_abs

#Declares the subgradient of the absolute value
def subgradient_abs_stoc(x_single):
    if x_single > 0:
        return 1
    if x_single < 0:
        return -1
 
    return random.uniform(-1, 1)
#end of subgradient_abs_stoc        
    
           
#Proximity function
#View hand-in for construction
def prox(t,x):
    
    def coordinate_prox(x_i):
        if x_i > lambda_value*t:
            return x_i - lambda_value*t
        if x_i < - lambda_value*t:
            return x_i + lambda_value*t
            
        return 0    
    
    return np.matrix(map(lambda k: coordinate_prox(k[0,0]), x.T))    
#end of prox

#declares the gloal constant alpha function
def alpha_fun_cons(x,g,a):
    return global_alpha
#end of alpha_fun_dec


#declares the gloal decreasing alpha function
def alpha_fun_decs(x,g,a):
    if a == None:
        return 1/alpha_step
    
    return 1/(1/a + alpha_step)
#end of alpha_fun_dec

def alpha_fun_back_prox(x,g,a):    
    
    rho = 1/2
    t = 1
    G = (x - prox(t,x - t*g))/t

    while(F(x - t*G) > F(x) - t*np.dot(G, g.T)[0,0] + (t/2)*(np.linalg.norm(G))**2):        
        t = rho*t
        G = (x - prox(t,x - t*g))/t
    
    return t
# end of alpha_backtracking

def alpha_fun_back_prox_acc(x,g,v):    
    
    rho = 4/5            
    t = 1
    
    while(F(x) > F(v) + np.dot((x-v),g.T)[0,0] + (1/(2*t))*(np.linalg.norm(x-v))**2):
        t = rho*t
        
    
    return t 
# end of alpha_backtracking_acc

#Method that constructs the corresponding backtracking method given the function and
# its gradient
def construct_apha_back(fun, fun_grad):

    def alpha_temp(x,g,a):
        #For the first iteration
        if(g is None):
            return global_alpha

        rho = 4/5
        c = 3/5        
        a = 1
        while(fun(x + a*g) > fun(x) + c*a*np.dot(fun_grad(x),g.T) ):
            a = rho*a
        
        return a

    return alpha_temp
#construct_apha_back    

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

def F_gradient_stoc(beta):
    
    index = random.sample(range(n), int(n*stochastic_percentage))

    data_temp = data_x[index,:]

    x_beta = np.dot(data_temp, beta.T)

    #first constructs the vector y_i + exp()/(1 + exp)
    #Constructs the gradient
    temp_vec = (-1)*(data_y[index,:]).T + map(lambda k: exp_over_exp(k[0,0]), x_beta)
    first_term = np.dot(temp_vec, data_temp)
        
    return(first_term)
#end of F_gradient_stoc

#Since f is differentiable, its subgradient is the gradient
def F_subgradient(beta):
	return F_gradient(beta)
#end of F_subgrdient

#Since f is differentiable, its subgradient is the gradient
def F_subgradient_stoc(beta):
    return F_gradient_stoc(beta)
#end of F_subgrdient    	

#declares the function g(x) = lamnda*|beta|
def G(beta):
	return lambda_value*np.linalg.norm(beta.T, 1)
#end of G

#Declares the subgradient of the fuction G
def G_subgradient(x_vec):    
    return lambda_value*np.array(map(subgradient_abs, x_vec.T)).T
#end of G_subgradient

#Declares the subgradient of the fuction G stochastic
def G_subgradient_stoc(x_vec):    
    return lambda_value*np.array(map(subgradient_abs_stoc, x_vec.T)).T
#end of G_subgradient_stoc

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
    return(F_gradient_stoc(beta) + G_subgradient_stoc(beta))
#end of H_subgradient_stoc


#----------------------------------------------------------------------
#-------------------------- Excecutions -------------------------------
#----------------------------------------------------------------------

#Defines the excecutions variables for the subgradient procedure
def excecute_subgradient(print_progress = True):
    return run_subgradient_descent(dim_data -1, 
                                   H, 
                                   H_subgradient, 
                                   alpha_fun_decs, 
                                   global_eps, 
                                   initial = None, 
                                   print_progress = print_progress)
#end of excecute_subgradient

#Defines the excecutions varaibles for the stochastic subgradient procedure
def excecute_subgradient_stoc(print_progress = True):
    return run_subgradient_descent(dim_data -1, 
                                   H,
                                   H_subgradient_stoc, 
                                   alpha_fun_decs, 
                                   global_eps, 
                                   initial = None, 
                                   print_progress = print_progress)
#edn of excecute_subgradient_stoc

#Defines the excecution variables for the proximal gradient procedure 
def excecute_proximal(print_progress = True):
    return run_proximal(dim_data-1, 
                        H, 
                        prox, 
                        F_gradient, 
                        alpha_fun_back_prox, 
                        global_eps, 
                        initial = None , 
                        print_progress = print_progress)
#end of excecute_proximal    

#Defines the excecution variables for the proximal accelerated procedure
def excecute_proximal_accelerated(print_progress = True):
    return run_proximal_accelerated(dim_data-1, 
                                    H, 
                                    prox, 
                                    F_gradient, 
                                    alpha_fun_cons, 
                                    global_eps, 
                                    initial = None , 
                                    print_progress = print_progress)
#end of excecute_proximal_accelerated    

#Defines the excecution variales for the proximal ADMM procedure
def excecute_ADDM(print_progress = True):

    def construct_f(i):
        def f_i(beta):
            x_beta = np.dot(data_x[i,:],beta.T)[0,0]
            return (-1)*data_y[i,0]*x_beta + log_exp(x_beta)

        return f_i  

    
    def construct_grad_f(i):
        def subgrad_f_i(beta):
            x_beta = np.dot(data_x[i,:],beta.T)[0,0]
            return ((-1)*data_y[i,0] + exp_over_exp(x_beta))*data_x[i,:]

        return subgrad_f_i  

    array_f = map(construct_f, range(n))
    grad_array_f = map(construct_grad_f, range(n))

    return  run_ADMM(dim_data - 1, 
                     array_f, 
                     grad_array_f, 
                     G, 
                     G_subgradient, 
                     alpha_fun_cons, 
                     global_eps, 
                     initial = None, 
                     print_progress = print_progress,
                     parallel = False)
#end of excecute_ADDM


'''

x = np.zeros((1,3))
x[0,0] = 1
x[0,1] = 1
x[0,2] = 1

print F_gradient(x)

'''
'''
resul = excecute_subgradient()

print resul[1]
print resul[4]
print resul[5]

sys.exit('Ok')

'''

#----------------------------------------------------------------------
#-------------------------- Graphing Script ---------------------------
#---------------------------------------------------------------------- 

'''
#Corre addm
r_ADMM =  excecute_ADDM(True)
print('Ok')
print('Numero de Iteraciones: ' + str(r_ADMM[4]))
print('Tiempo: ' + str(r_ADMM[5]))
print('Minimo: ' + str(r_ADMM[1]))
print('------------------------------')

dif = map(lambda y: math.log(math.fabs(y - r_ADMM[1])),(t for t in r_ADMM[3] if t > r_ADMM[1]))
trace_1 = go.Scatter(x = range(len(dif)), y =  dif)
plot_url = py.plot([trace_1], auto_open=False)


sys.exit('Ok')
'''
'''
#Runs the main experiment for each method and the graphs it
print 'Start Subgradient '
r_sub =  excecute_subgradient(False)
print('Ok')
print('Numero de Iteraciones: ' + str(r_sub[4]))
print('Tiempo: ' + str(r_sub[5]))
print('Minimo: ' + str(r_sub[1]))
print('------------------------------')
print('')
print('------------------------------')
print('Start Stocastic Subgradient')
r_stoc =  excecute_subgradient_stoc(False)
print('Ok')
print('Numero de Iteraciones: ' + str(r_stoc[4]))
print('Tiempo: ' + str(r_stoc[5]))
print('Minimo: ' + str(r_stoc[1]))
print('------------------------------')
print('')
print('------------------------------')
print('Start Proximal')
r_prox =  excecute_proximal(False)
print('Ok')
print('Numero de Iteraciones: ' + str(r_prox[4]))
print('Tiempo: ' + str(r_prox[5]))
print('Minimo: ' + str(r_prox[1]))
print('------------------------------')
print('')
print('------------------------------')

print('Start Proximal Accelerated')
r_acc =  excecute_proximal_accelerated(False)
print('Ok')
print('Numero de Iteraciones: ' + str(r_acc[4]))
print('Tiempo: ' + str(r_acc[5]))
print('Minimo: ' + str(r_acc[1]))
print('------------------------------')
print('')
print('------------------------------')



#Plots the results
#plot_log(resultado[3], resultado[1])
#Graphs the plot for log
dif = map(lambda y: math.log(math.fabs(y - r_sub[1])),(t for t in r_sub[3] if t > r_sub[1]))
trace_1 = go.Scatter(x = range(len(dif)), y =  dif)
dif = map(lambda y: math.log(math.fabs(y - r_stoc[1])), (t for t in r_stoc[3] if t > r_stoc[1]))
trace_2 = go.Scatter(x = range(len(dif)), y =  dif)
dif = map(lambda y: math.log(math.fabs(y - r_prox[1])),(t for t in r_prox[3] if t > r_prox[1]))
trace_3 = go.Scatter(x = range(len(dif)), y =  dif)
dif = map(lambda y: math.log(y - r_acc[1] ),(t for t in r_acc[3] if t > r_acc[1]))
trace_4 = go.Scatter(x = range(len(dif)), y =  dif)

#Export graph
plot_url = py.plot([trace_1,trace_2,trace_3, trace_4], auto_open=False)

print('Grafica logaritmica hecha')

'''

#Starts the experient for different lambda

res_sub = []
res_stoc = []
res_prox = []
res_acc = []

lambdas = [0.5] + [1]  #range(1,1001,263) 

for l in lambdas:
    lambda_value = l
    r_sub =  excecute_subgradient(print_progress = False)
    res_sub.append([r_sub[0],r_sub[1],r_sub[4],r_sub[5]])

    r_stoc =  excecute_subgradient_stoc(print_progress = False)
    res_stoc.append([r_stoc[0],r_stoc[1],r_stoc[4],r_stoc[5]])

    r_prox =  excecute_proximal(print_progress = False)
    res_prox.append([r_prox[0],r_prox[1],r_prox[4],r_prox[5]])

    r_acc =  excecute_proximal_accelerated(print_progress = False)
    res_acc.append([r_acc[0],r_acc[1],r_acc[4],r_acc[5]])

    print ('Finished: ' + str(l))

#Now plots the different graphs
#Norm of the B^*
trace_1 = go.Scatter(x = lambdas, y =  map(lambda w: np.linalg.norm(w[0].T,1), res_sub))
trace_2 = go.Scatter(x = lambdas, y =  map(lambda w: np.linalg.norm(w[0].T,1), res_stoc))
trace_3 = go.Scatter(x = lambdas, y =  map(lambda w: np.linalg.norm(w[0].T,1), res_prox))
trace_4 = go.Scatter(x = lambdas, y =  map(lambda w: np.linalg.norm(w[0].T,1), res_acc))
plot_url = py.plot([trace_1,trace_2,trace_3,trace_4], auto_open=False)
print 'Grafica Tamanho de Beta'

#Minimum reached
trace_1 = go.Scatter(x = lambdas, y =  map(lambda w: w[1], res_sub))
trace_2 = go.Scatter(x = lambdas, y =  map(lambda w: w[1], res_stoc))
trace_3 = go.Scatter(x = lambdas, y =  map(lambda w: w[1], res_prox))
trace_4 = go.Scatter(x = lambdas, y =  map(lambda w: w[1], res_acc))
plot_url = py.plot([trace_1,trace_2,trace_3,trace_4], auto_open=False)
print 'Grafica Minimo Alcanzado'

#Number of Iterations
trace_1 = go.Scatter(x = lambdas, y =  map(lambda w: w[2], res_sub))
trace_2 = go.Scatter(x = lambdas, y =  map(lambda w: w[2], res_stoc))
trace_3 = go.Scatter(x = lambdas, y =  map(lambda w: w[2], res_prox))
trace_4 = go.Scatter(x = lambdas, y =  map(lambda w: w[2], res_acc))
plot_url = py.plot([trace_1,trace_2,trace_3,trace_4], auto_open=False)
print 'Grafica Numero de Iteraciones'


#Time required
trace_1 = go.Scatter(x = lambdas, y =  map(lambda w: w[3], res_sub))
trace_2 = go.Scatter(x = lambdas, y =  map(lambda w: w[3], res_stoc))
trace_3 = go.Scatter(x = lambdas, y =  map(lambda w: w[3], res_prox))
trace_4 = go.Scatter(x = lambdas, y =  map(lambda w: w[3], res_acc))
plot_url = py.plot([trace_1,trace_2,trace_3,trace_4], auto_open=False)
print 'Grafica Tiempo requerido'


print 'Done!'











