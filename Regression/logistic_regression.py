import numpy as np
import pandas as pd

from scipy.stats import norm
import random


class logistic_regression:
    
    def __init__(self, X, y):
        '''
        Class for estimating logistic regression coefficients which
        are trained using explicit gradient descent on the log-likelihood
        function to find max likelihood estimates
        
        Parameters
        ----------
        X : Data stored in a numpy ndarray of shape - (observations x variables)
        y : Score for each observation (0,1); y has one column with the same
            number of rows as X
            
        Class Variables
        ---------------
        self.X and self.Y : X and y
        self.intercept    : bool that allows for the inclusion or exclusion of an 
                            intercept, defaults to true
        self.variance     : numpy ndarray that stores variance for each observation
                            in our model
        self.neg_hessian  : stores the negative hessian for our MLE, this is 
                            used to find the fisher information matrix containing
                            the standard error for our coefficients
        self.probabilities: stores the probability of observing each observation
                            in our model
        self.coefficients : dictionary that stores our coefficients
        '''
        self.X = X
        self.y = y.reshape((X.shape[0],1))
        self.intercept = True
        self.variance = np.zeros((X.shape[0],1))
        self.neg_hessian = np.zeros((X.shape[1]+1, X.shape[1]+1))
        self.probabilities = np.zeros((X.shape[0],1))
        self.coefficients = {}
        for c in range(X.shape[1] + 1):
            self.coefficients["b" + str(c)] = random.uniform(-0.01, 0.01)
            
            
    # Returns current coefficients without intercept
    def get_coefficients(self):
        coef = []
        for c in range(len(self.coefficients) - 1):
            coef.append(self.coefficients["b" + str(c + 1)])
                
        return coef
        
    # Sets coefficients to values in input list 
    def set_coefficients(self, list_c):
        
        if self.intercept == True:
            if len(list_c) == len(self.coefficients):
                for i, num in enumerate(list_c):
                    self.coefficients["b" + str(i)] = num
            else:
                print("Did not set coefficients, input list did not match number of parameters")
        else:
            if len(list_c) == len(self.coefficients) - 1:
                for i, num in enumerate(list_c):
                    self.coefficients["b" + str(i+1)] = num
            else:
                print("Did not set coefficients, input list did not match number of parameters")
                
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
    # If l = our local function at each obeservation, and y the score at each obs, 
    # the proability of observing each observation in our current model is expressed as: 
    # (e^l/(1+e^l))^y * (1/(1+e^l))^(1-y)
    # Our log-likelihood result is found by taking the natural log of each probability
    # and summing these results       
    def compute_log_likelihood(self):
        self.probabilities = ((np.exp(self.loc_func())/(1 + np.exp(self.loc_func())))**self.y)*((1/(1 + np.exp(self.loc_func())))**(1-self.y))
    
        return np.sum(np.log(self.probabilities))

    # Defines a step along the gradient of our log-likelihood function
    def step_gradient(self, learning_rate):
        c = np.array(self.get_coefficients())
        # Compute intercept gradient - derivation in documents
        b0_grad = np.sum(self.y - np.exp(self.loc_func())/(1 + np.exp(self.loc_func())))
        # Compute variable coefficient gradients - derivation in documents
        bc_grad = np.sum(np.multiply(self.y - np.exp(self.loc_func())/(1 + np.exp(self.loc_func())), self.X), axis = 0)
        # Increment coefficient values along gradient
        new_b0 = self.coefficients["b0"] + (learning_rate * b0_grad)
        new_bc = c + (learning_rate * bc_grad)
        
        # Parameter update
        if self.intercept == True:
            paramater_update = [new_b0]
        else:
            paramater_update = []

        for new_c in new_bc:
            paramater_update.append(new_c)
          
        self.set_coefficients(paramater_update)
        
        # Compute new log-likelihood
        new_log = self.compute_log_likelihood()
        
        # Compute total gradient
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
        # Compute variances for each observation
        self.variance = np.multiply(self.probabilities, (1 - self.probabilities))
        
        if self.intercept == True:
            temp_X = np.insert(self.X, 0, 1, axis = 1)
        else:
            temp_X = self.X
        
        # Compute negative hessian
        self.neg_hessian = np.dot(np.multiply(self.variance.T, temp_X.T), temp_X)
        
        # Find standard error for coefficients from the inversion of neg_hessian
        # square root along the diagonal
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
        
        # Calculate p-values
        p_value = [2*(1-norm.cdf(abs(bc[i])/std_error[i])) for i in range(len(std_error))]
        
        # Printing nicely        
        final_nda = np.array([bc, std_error, p_value, L_95, U_95])
        display = pd.DataFrame(final_nda, index = ["Estimates", "Standard Error", "p-value", L_ci, U_ci], columns = ["b" + str(c) if self.intercept==True else "b" + str(c+1) for c in range(len(bc))])
        
        print(display)
    
    # Once model is trained, provide variable values to predict score/probability 
    # of success
    def predict_prob(self, variables):
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