from Regression.logistic_regression import logistic_regression as lr
import numpy as np
import pandas as pd
from scipy.stats import norm
import random

 
 ###########################################################################
 ## Data is on blood donation, includes features:                         ##
 ##     Recency of last visit in months                                   ##
 ##     Frequency - number of times an observation has given previously   ##
 ##     Time - In months, since last donation                             ##
 ##                                                                       ##
 ## Data includes one objective variable:                                 ##
 ##     Whether he or she donated blood in March 2007                     ##
 ##                                                                       ##
 ## We will build a logistic regression model that predicts the           ##
 ## probability of an individual donating blood given their donation      ##
 ## history.                                                              ##
 ##                                                                       ##
 ###########################################################################
 

# Load Data
data = np.genfromtxt("csvs/BloodDonation.csv", delimiter=",")
# Deletes column names
if np.isnan(data[0,0]):
    data = np.delete(data, 0, 0)
    
# Inspect data - checking to make sure dimensions are correct (748,4)
data.shape

# Save X matrix of features and y vector of objective scores
X = data[:,0:3].reshape(-1,3)
X.shape

y = data[:,3]

# Begin regression instance with data X and y
ln = lr(X,y)

# Check beginning log-likelihood
ln.compute_log_likelihood()

# Start gradient descent, first I check to make sure total gradient is decreasing with each step, 
# learning rate was initially set at 0.0001, this turned out to be too large, gradient was overstepping 0
# set new gradient at 0.00001
lr = 0.00001
ln.step_gradient(lr)

# Performing gradient descent loop 1000 times at first 
# gradient_descent takes in the learning rate and number of iterations,
# Returns the sum of current weight gradients and the current log-likelihood
ln.gradient_descent(lr, 1000)

# After 89,000 total steps, reduced sum of gradients = -1.399e-11
# Compute maximized log-likelihood
ln.compute_log_likelihood() # = -353.933


# Print out coefficients with statistics
#                       b0            b1            b2        b3
# Estimates      -0.449540 -9.858382e-02  1.353895e-01 -0.023092
# Standard Error  0.180349  1.731709e-02  2.567216e-02  0.005964
# p-value         0.012680  1.249041e-08  1.336353e-07  0.000108
# Lower 95% CI   -0.803023 -1.325253e-01  8.507210e-02 -0.034782
# Upper 95% CI   -0.096056 -6.464233e-02  1.857069e-01 -0.011403
ln.get_statistics()

# Print inverse of neg_hessian, this matrix includes the fisher information
#   [  0.032526 -0.001645 -0.000990 -0.000249 ]
#   [ -0.001645  0.000300  0.000095 -0.000026 ]
#   [ -0.000990  0.000095  0.000659 -0.000114 ]
#   [ -0.000249 -0.000026 -0.000114  0.000036 ]
print(pd.DataFrame(np.linalg.inv(ln.neg_hessian)))


# Predict score for individual based on their donation history
ln.predict_prob([5, 10, 24])



