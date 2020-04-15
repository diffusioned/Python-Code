# Multiprocessing and Scipy NLLS Regression.py by Ned Charles, April 2020
# A Python script that compares NLLS Regression fitting using Scipy least_squares in a standard for loop versus parallel processing
# with the multiprocessing Pool capabilities
# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
# and: https://docs.python.org/3.8/library/multiprocessing.html
#############################################################################################################
# Import needed Python math and fitting modules
import numpy as np
from scipy.optimize import least_squares
import time
# Plotting module
import matplotlib.pyplot as plt
# Multiprocesing module
from multiprocessing import Pool
# Iteration tools module
from itertools import starmap
#############################################################################################################
# Define the objective function for the model to use with curve_fit.
def fcn2minExpCos(x, beta1, beta2):
    return np.exp(-beta1*x) * np.cos(beta2*x)   # Exponential decay times a cosine function
#############################################################################################################
# Define the objective function for the model to use with least_squares.
def fcn2minExpCosErrFunc(beta, x, y):
    return (y - (np.exp(-beta[0]*x) * np.cos(beta[1]*x)))  # Subtract the model signal from the actual signal
#############################################################################################################
# Define the Jacobian function for the model to use with least_squares.
def jac2minExpCosErrFunc(beta, x, y=None):
    # Model function =  np.exp(-beta1*x) * np.cos(beta2*x)
    JacobPt1 = -x * np.exp(-beta[0]*x) * np.cos(beta[1]*x) # d/da(exp(-a*x)*cos(b*x))
    JacobPt2 = -x * np.exp(-beta[0]*x) * np.sin(beta[1]*x) # d/db(exp(-a*x)*cos(b*x))
    return np.transpose([JacobPt1, JacobPt2])
#############################################################################################################
# Simulate data using the same function we will fit to.
x = np.linspace(0, 10.0, num=101)           # Generate array of 101 data points from zero to ten in 0.1 increments
Beta1 = 0.5                                 # First Beta parameter for the exponential decay portion
Beta2 = 5                                   # Second Beta parameter for the cosine portion
NumParams = 2                               # Set the value based on the number of model parameters 
StdNoise = 0.1                              # Set the value for the standard deviation of the noise
yNoiseFree = fcn2minExpCos(x, Beta1, Beta2) # Generate the "signal" data before adding noise

# Add normally distributed noise multiple times to create a set of noisy data
NumberOfNoisyDataSamples = 10000
NoisyDataSamples = np.zeros((NumberOfNoisyDataSamples, len(x)))
for i in range(0, NumberOfNoisyDataSamples):
    RandomNoiseSamples = np.random.normal(size=len(x), scale=StdNoise)
    NoisyDataSamples[i, :] = yNoiseFree + RandomNoiseSamples

# Optional code to plot the original signal and overlay noisy data to show the scale of the noise
# plt.plot(x, y, 'b')
# plt.plot(x, yNoisy, 'r')
# plt.xlabel('x')
# plt.ylabel('y (blue) and yNoise (red)')
# plt.show()
#############################################################################################################
# To prepare for fitting multiple noisy data sets, declare everything that doesn't change inside the fitting loop
InitialParams = [1., 1.]
# Also declare all arrays to store the saved data in, so they can be examined after
Beta1ParamArray = np.zeros(NumberOfNoisyDataSamples)
Beta2ParamArray = np.zeros(NumberOfNoisyDataSamples)
# Parameter array for using multiprocessing Pool functionality
Beta1PoolParamArray = np.zeros(NumberOfNoisyDataSamples)
Beta2PoolParamArray = np.zeros(NumberOfNoisyDataSamples)
# Other optional fitting data to save
# ResidualArray = np.zeros((NumberOfNoisyDataSamples, len(x)))
# JacobianArray = np.zeros((NumberOfNoisyDataSamples, len(x), NumParams))

#############################################################################################################
# Standard for loop
startFitting = time.time()
for i in range(0, NumberOfNoisyDataSamples):
    NoisyDataSet = NoisyDataSamples[i, :]
    # Option - fit without Jacobian
    LSOptimResult = least_squares(fcn2minExpCosErrFunc, InitialParams, method='lm', args=(x, NoisyDataSet), verbose=0)
    # Option - fit with Jacobian
    # LSOptimResult = least_squares(fcn2minExpCosErrFunc, InitialParams, jac=jac2minExpCosErrFunc, method='lm', args=(x, NoisyDataSet), verbose=0)
    Beta1ParamArray[i] = LSOptimResult.x[0]
    Beta2ParamArray[i] = LSOptimResult.x[1]
    # Optional fitting metrics from algorithm
    # ResidualArray[i,:] = LSOptimResult.fun
    # JacobianArray[i,:] = LSOptimResult.jac

endFitting = time.time()
print('Total Fitting Time: ' + str(endFitting - startFitting))
#############################################################################################################
# Multiprocessing pool
def FitNoisyData(x, NoisyDataSet):
    # Option - fit without Jacobian
    LSOptimResult = least_squares(fcn2minExpCosErrFunc, InitialParams, method='lm', args=(x, NoisyDataSet), verbose=0)
    # Option - fit with Jacobian
    # LSOptimResult = least_squares(fcn2minExpCosErrFunc, InitialParams, jac=jac2minExpCosErrFunc, method='lm', args=(x, NoisyDataSet), verbose=0)
    return [LSOptimResult.x[0], LSOptimResult.x[1]]

# Add this line for Pool to work properly
if __name__ == '__main__':
    startFitting = time.time()
    # Tile the x values multiple times to match the data array size
    xTiled = np.tile(x, (NumberOfNoisyDataSamples, 1))
    # Setup multiprocessing pool
    pool = Pool(4)
    ResultList = pool.starmap(FitNoisyData, zip(xTiled, NoisyDataSamples)) # Use zip to interlace the x and y values
    pool.close() 
    pool.join()
    # Data gets returned as a list, convert to array to reshape for the parameter arrays
    ResultArray = np.array(ResultList)
    Beta1PoolParamArray = ResultArray[:,0]
    Beta2PoolParamArray = ResultArray[:,1]

    endFitting = time.time()
    print('Total Fitting Time: ' + str(endFitting - startFitting))
#############################################################################################################
# # Python itertools starmap function
# startFitting = time.time()
# # Tile the x values multiple tiles to match the data
# xTiled = np.tile(x, (NumberOfNoisyDataSamples, 1))
# ResultList = list(starmap(FitNoisyData, zip(xTiled, NoisyDataSamples))) # Use zip to interlace the x and y values
# ResultArray = np.array(ResultList)
# Beta1ParamArray = ResultArray[:,0]
# Beta2ParamArray = ResultArray[:,1]
# endFitting = time.time()
# print('Total Fitting Time: ' + str(endFitting - startFitting))
#############################################################################################################
# Post-fitting data analysis if you want to analyze other fitting measures
# # Calculate the RSS value based on the squared residuals
# RSS = sum(LSOptimResult.fun**2)
# # Calculate the SER (Standard Error of Regression)
# NumMeas = len(x)   # number of data points
# SER = np.sqrt(RSS/(NumMeas - NumParams))
# # Compare this to the standard deviation of the residuals
# StdResiduals = np.std(LSOptimResult.fun)
# # Calculate the covariance matrix based on the returned Jacobian using QR decomposition
# q, r = np.linalg.qr(LSOptimResult.jac)                  # Calculate the Q & R values of the Jacobian matrix
# rInv = np.linalg.solve(r, np.identity(r.shape[0]))      # Calcuate the inverse value of R
# JTJInv = np.matmul(rInv, rInv.transpose())              # Matrix multiply R Inverse by the transpose of R Inverse
# CovMatrix = SER**2 * JTJInv                             # Multiply this matrix by the squared regression error (the variance)

# Optional - display a histogram of the residuals
# plt.figure()
# hist, bins = np.histogram(LSOptimResult.fun, bins=25)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.show()
