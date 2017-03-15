import numpy as np
import pandas as pd

from scipy.stats import norm
import random


class binom_regression:
     '''
        Class for estimating binary regression coefficients which
        are trained using explicit gradient descent on the log-likelihood
        function to find max likelihood estimates
        
        Parameters
        ----------
        X : Data stored in a numpy ndarray of shape - (observations x variables)
        k : Number of successes in n events for each observation; k has one column
            with the same number of rows as X 
        n : Number of events for each observation; n has one column with the same
            number of rows as X
            
        Class Variables
        ---------------
        self.X, self.k/n  : X, k, and n
        self.intercept    : bool that allows for the inclusion or exclusion of an 
                            intercept, defaults to true
        self.variance     : numpy ndarray that stores variance for each observation
                            in our model
        self.neg_hessian  : stores the negative hessian for our MLE, this is 
                            used to find the fisher information matrix containing
                            the standard error for our coefficients
        self.p            : local p values in each observation's binomial function
        self.probabilities: stores the probability of observing each observation
                            in our model
        self.coefficients : dictionary that stores our coefficients
        '''
    def __init__(self, X, k, n):
        self.X = X
        self.k = k.reshape(-1,1)
        self.n = n.reshape(-1,1)
        self.intercept = True
        self.variance = np.zeros((X.shape[0],1))
        self.neg_hessian = np.zeros((X.shape[1]+1, X.shape[1]+1))
        self.p = np.zeros((X.shape[0],1))
        self.probabilities = np.zeros((X.shape[0],1))
        self.coefficients = {}
        for c in range(X.shape[1] + 1):
            self.coefficients["b" + str(c)] = random.uniform(-0.01, 0.01)
    
    # Binomial function for use in calculating local probabilities
    def binom_function(self, k, n, p):
        result = (np.array([np.math.factorial(i) for i in n])/np.multiply(np.array([np.math.factorial(j) for j in k]), np.array([np.math.factorial(l) for l in (n-k)]))).reshape(p.shape)*p**k*(1-p)**(n-k)
        return result
    
    # Returns current coefficients without intercept        
    def get_coefficients(self):
        coef = []
        for c in range(len(self.coefficients) - 1):
            coef.append(self.coefficients["b" + str(c + 1)])
                
        return coef
    
    # Sets coefficients to values in input list     
    def set_coefficients(self, list_c):
        
        if self.intercept == True:
            for i, num in enumerate(list_c):
                self.coefficients["b" + str(i)] = num
        else:
            for i, num in enumerate(list_c):
                self.coefficients["b" + str(i+1)] = num
    
    # If called, sets self.intercept to False                               
    def no_int(self):
        self.intercept = False
        
    # Helper function to make long equations more readable, local function is 
    # b0 + bn*Xn
    def loc_func(self):
        c = np.array(self.get_coefficients())
        
        if self.intercept == True:
            return self.coefficients["b0"] + np.sum(c*self.X, axis = 1).reshape(-1,1)
            
        else:
            return np.sum(c*self.X, axis = 1).reshape(-1,1)

    # Computes the log_likelihood of our model        
    def compute_log_likelihood(self):
        self.p = np.exp(self.loc_func())/(1 + np.exp(self.loc_func()))
        
        self.probabilities = self.binom_function(self.k, self.n, self.p)
    
        return np.sum(np.log(self.probabilities))

    # Defines a step along the gradient of our log-likelihood function
    def step_gradient(self, learning_rate):
        c = np.array(self.get_coefficients())
        df_dp = self.k/self.p - (self.n - self.k)/(1-self.p)
        dp_db0 = np.exp(self.loc_func())/((1 + np.exp(self.loc_func()))**2)
        dp_dbX = np.multiply(np.exp(self.loc_func())/((1 + np.exp(self.loc_func()))**2), self.X)
        b0_grad = np.sum(np.multiply(df_dp, dp_db0))
        bc_grad = np.sum(np.multiply(df_dp, dp_dbX), axis = 0)
        new_b0 = self.coefficients["b0"] + (learning_rate * b0_grad)
        new_bc = c + (learning_rate * bc_grad)
        
        if self.intercept == True:
            paramater_update = [new_b0]
        else:
            paramater_update = []

        for new_c in new_bc:
            paramater_update.append(new_c)
          
        self.set_coefficients(paramater_update)
        new_log = self.compute_log_likelihood()
        
        if self.intercept == True:
            return [np.sum(bc_grad) + b0_grad, new_log]
        else:
            return [np.sum(bc_grad), new_log]
        
    # Gradient descent function that takes in parameters for learning rate
    # and the number of iterations to step, prints out new log-likelihood
    # and total gradient for the last step taken                
    def gradient_descent(self, learning_rate, num_iterations):
        for i in range(num_iterations):
            grad_prog = self.step_gradient(learning_rate)
            
        print("Total Grad: " + str(grad_prog[0]) + '\n' + "New Log-Likelihood: " + str(grad_prog[1]))
    
    # Function that prints out statistics for the model, these include:
    # p-values, standard error, and confidence intervals for coefficients
    # confidence interval is defaulted to 95%, but can be altered to 90%
    def get_statistics(self, CI = 95):
        self.compute_log_likelihood()
        self.variance = np.multiply(self.p, (1 - self.p))*self.n
        
        if self.intercept == True:
            temp_X = np.insert(self.X, 0, 1, axis = 1)
        else:
            temp_X = self.X
        
        self.neg_hessian = np.dot(np.multiply(self.variance.T, temp_X.T), temp_X)
        
        std_error = [np.sqrt(np.linalg.inv(self.neg_hessian))[x,x] for x in range(self.neg_hessian.shape[0])]
        
                     
        if self.intercept == True:
            bc = [self.coefficients["b0"]] + self.get_coefficients() 
        else:
            bc = self.get_coefficients()
        
        if CI == 95:
            L_95 = np.array(bc) - (1.96*np.array(std_error))
            U_95 = np.array(bc) + (1.96*np.array(std_error))
            L_ci = "Lower 95% CI"
            U_ci = "Upper 95% CI"
            
        elif CI == 90:
            L_95 = np.array(bc) - (1.645*np.array(std_error))
            U_95 = np.array(bc) + (1.645*np.array(std_error))
            L_ci = "Lower 90% CI"
            U_ci = "Upper 90% CI"
            
        else:
            L_95 = 'na'
            U_95 = 'na'
            L_ci = "Lower CI"
            U_ci = "Upper CI"
        
        p_value = [2*(1-norm.cdf(abs(bc[i])/std_error[i])) for i in range(len(std_error))]
                   
        final_nda = np.array([bc, std_error, p_value, U_95, L_95])
        display = pd.DataFrame(final_nda, index = ["Estimates", "Standard Error", "p-value", U_ci, L_ci], columns = ["b" + str(c) if self.intercept==True else "b" + str(c+1) for c in range(len(bc))])
        
        print(display)
    
    # Print the final formula
    def print_formula(self):
        if self.intercept == True:
            local_string = "{0:.2f} + ".format(self.coefficients['b0']) 
        else:
            local_string = ""
            
        for i, y in enumerate(self.get_coefficients()):
            local_string += "{0:.4f}*x{1}_i + ".format(y, i+1)
        local_string = local_string[0:-3]
        
        print("e^(" + local_string + ")/(1 + e^(" + local_string + "))")
    
    # Once model is trained, provide variable values to predict p value for 
    # that observation, meaning the probability of success on each try for 
    # that particular observation
    def predict_p(self, variables):
        Xi = np.array(variables)
        c = np.array(self.get_coefficients())
        if len(c) == len(Xi):
            local_sum = np.sum(c*Xi)
            if self.intercept == True:
                local_sum += self.coefficients["b0"]
            return np.exp(local_sum)/(1+np.exp(local_sum))
        else:
            print("Incorrect number of variables for this model")
            return