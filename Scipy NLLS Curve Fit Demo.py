# Scipy NLLS Curve Fit Demo.py by Ned Charles, February 2020
# An example Python script that walks through how to do a nonlinear, least squares (NLLS) regression fit on simulated data.
# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit
#############################################################################################################
# Import needed Python math and fitting modules
import numpy as np
from scipy.optimize import curve_fit, least_squares
from lmfit import minimize, Minimizer, Parameters, Parameter, fit_report
import time

# Plotting modules
import matplotlib.pyplot as plt
# Uncomment for 3D surface plot at bottom
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

#############################################################################################################
# Define the objective function for the model to use with curve_fit.
def fcn2minExpCos(x, beta1, beta2):
    return np.exp(-beta1*x) * np.cos(beta2*x)   # Exponential decay times a cosine function
#############################################################################################################
# Define the objective function for the model to use with least_squares.
def fcn2minExpCosErrFunc(beta, x, y):
    return (y - (np.exp(-beta[0]*x) * np.cos(beta[1]*x)))  # Subtract the model signal from the actual signal
#############################################################################################################
# Define the objective function for the model to use with least_squares.
def fcn2minLMExpCosErrFunc(params, x, y):
    beta1 = params['beta1']
    beta2 = params['beta2']
    return (y - (np.exp(-beta1*x) * np.cos(beta2*x)))  # Subtract the model signal from the actual signal
#############################################################################################################
# SymPy calculate the Jacobian for the model function
# from sympy import *
# b1, b2, x = symbols('b1 b2 x')
# init_printing(use_unicode=True)
# Modeldb1 = diff((exp(-b1*x) * cos(b2*x)), b1)
# Modeldb2 = diff((exp(-b1*x) * cos(b2*x)), b2)
# print(Modeldb1)
# print(Modeldb2)
#############################################################################################################
# Define the Jacobian function for the model to use with curve_fit.
def jac2minExpCos(x, beta1, beta2):
    # Model function =  np.exp(-beta1*x) * np.cos(beta2*x)
    JacobPt1 = -x * np.exp(-beta1*x) * np.cos(beta2*x) # d/da(exp(-a*x)*cos(b*x))
    JacobPt2 = -x * np.exp(-beta1*x) * np.sin(beta2*x) # d/db(exp(-a*x)*cos(b*x))
    return np.transpose([JacobPt1, JacobPt2])
#############################################################################################################
# Define the Jacobian function for the model to use with least_squares.
def jac2minExpCosErrFunc(beta, x, y=None):
    # Model function =  np.exp(-beta1*x) * np.cos(beta2*x)
    JacobPt1 = -x * np.exp(-beta[0]*x) * np.cos(beta[1]*x) # d/da(exp(-a*x)*cos(b*x))
    JacobPt2 = -x * np.exp(-beta[0]*x) * np.sin(beta[1]*x) # d/db(exp(-a*x)*cos(b*x))
    return np.transpose([JacobPt1, JacobPt2])
#############################################################################################################
# Define the Jacobian function for the model to use with LMFit.
def jac2minLMExpCosErrFunc(params, x, y):
    beta1 = params['beta1']
    beta2 = params['beta2']
    # Model function =  np.exp(-beta1*x) * np.cos(beta2*x)
    JacobPt1 = -x * np.exp(-beta1*x) * np.cos(beta2*x) # d/da(exp(-a*x)*cos(b*x))
    JacobPt2 = -x * np.exp(-beta1*x) * np.sin(beta2*x) # d/db(exp(-a*x)*cos(b*x))
    return np.transpose([JacobPt1, JacobPt2])
#############################################################################################################
# Simulate data using the same function we will fit to.
x = np.linspace(0, 10.0, num=101)   # Generate array of 101 data points from zero to ten in 0.1 increments
Beta1 = 0.5                         # First Beta parameter for the exponential decay portion
Beta2 = 5                           # Second Beta parameter for the cosine portion
NumParams = 2                       # Set the value based on the number of model parameters 
StdNoise = 0.1                      # Set the value for the standard deviation of the noise
y = fcn2minExpCos(x, Beta1, Beta2)  # Generate the signal values before adding noise

# Generate random noise sampled from a normal (Gaussian) distribution with the standard deviation specified
# above and a mean of zero.  The number of samples will equal the number of data points and the noise samples
# are added to the original signal
noiseSamples = np.random.normal(size=len(y), scale=StdNoise)
yNoisy = y + noiseSamples

# Plot the original signal and overlay the noisy signal to show the scale of the noise
# plt.plot(x, y, 'b')
# plt.plot(x, yNoisy, 'r')
# plt.xlabel('x')
# plt.ylabel('y (blue) and yNoise (red)')
# plt.show()
#############################################################################################################
# Now use the NLLS regression function curve_fit to fit the noisy data
# Set the initial parameter values (starting guess) for the regression algorithm
InitialParams = [1., 1.]
#############################################################################################################
# Fit the data with the SciPy curve_fit algorithm
# startCF = time.time()
fitParams, pcov = curve_fit(fcn2minExpCos, x, yNoisy, p0=InitialParams, method='lm') # no Jacobian
# fitParams, pcov = curve_fit(fcn2minExpCos, x, yNoisy, p0=InitialParams, method='lm', jac=jac2minExpCos)
# endCF = time.time()
# print('curve_fit: ' + str(endCF - startCF))
#############################################################################################################
# Or fit the function and date with the SciPy least_squares function
# startLS = time.time()
LSOptimResult = least_squares(fcn2minExpCosErrFunc, InitialParams, method='lm', args=(x, yNoisy), verbose=0)
# LSOptimResult = least_squares(fcn2minExpCosErrFunc, InitialParams, jac=jac2minExpCosErrFunc, method='lm', args=(x, yNoisy), verbose=0)
# endLS = time.time()
# print('least_squares: ' + str(endLS - startLS))

# Calculate the RSS value based on the squared residuals
RSS = sum(LSOptimResult.fun**2)
# Calculate the SER (Standard Error of Regression)
NumMeas = len(yNoisy)   # number of data points
SER = np.sqrt(RSS/(NumMeas - NumParams))
# Compare this to the standard deviation of the residuals
StdResiduals = np.std(LSOptimResult.fun)
# Calculate the covariance matrix based on the returned Jacobian using QR decomposition
q, r = np.linalg.qr(LSOptimResult.jac)                  # Calculate the Q & R values of the Jacobian matrix
rInv = np.linalg.solve(r, np.identity(r.shape[0]))      # Calcuate the inverse value of R
JTJInv = np.matmul(rInv, rInv.transpose())              # Matrix multiply R Inverse by the transpose of R Inverse
CovMatrix = SER**2 * JTJInv                             # Multiply this matrix by the squared regression error (the variance)

# Display a histogram of the residuals
# plt.figure()
# hist, bins = np.histogram(LSOptimResult.fun, bins=25)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.show()
#############################################################################################################
# Fit the data with the LMFit algorithm
# Define and initialize the parameters
LMparams = Parameters()
LMparams.add('beta1', value = 1.)
LMparams.add('beta2', value = 1.)

# Pass these parameter along with the data and independent variable into the Minimizer object

LMFitmin = Minimizer(fcn2minLMExpCosErrFunc, LMparams, fcn_args=(x, yNoisy))
# startLM = time.time()
LMFitResult = LMFitmin.minimize(method='leastsq')
# LMFitResult = LMFitmin.minimize(method='leastsq', Dfun=jac2minLMExpCosErrFunc)
# endLM = time.time()
# print('LMFit' + str(endLM - startLM))

#############################################################################################################
# Calculate the RSS values in the neighborhood of the parameter values
# Beta1Range = np.linspace(0, 7, num=141)                 # Set the Beta1 values from 0 to 7 at 0.05 intervals
# Beta2Range = np.linspace(0, 10, num=101)                # Set the Beta2 values from 0 to 10 at 0.1 intervals
# RSSForBetaParams = np.zeros((len(Beta1Range),len(Beta2Range)))
# for i in range(0, len(Beta1Range)):
#     for j in range(0, len(Beta2Range)):
#         RSSForBetaParams[i,j] = sum((fcn2minExpCos(x, Beta1Range[i], Beta2Range[j]) - yNoisy)**2)
# Beta1Range, Beta2Range = np.meshgrid(Beta2Range, Beta1Range)     # Reverse the order to match the RSS grids below

# Plot a 3D surface plot of the RSS values
# fig = plt.figure()
# ax3 = Axes3D(fig)                                      
# surf3 = ax3.plot_surface(Beta2Range, Beta1Range, RSSForBetaParams, cmap=cm.jet, vmin=0, vmax=20, rcount=141, ccount=101)
# ax3.set_zlim(0, 20)
# ax3.set_xlabel('β2')
# ax3.set_ylabel('β1')
# ax3.set_zlabel('RSS')
# fig.colorbar(surf3, shrink=0.5, aspect=5)
# plt.show()

# Or a 2D top down contour plot
# fig = plt.figure()
# ContourRange = np.linspace(0, 10, num=101) 
# cntr = plt.contour(Beta2Range, Beta1Range, RSSForBetaParams, ContourRange)
# plt.xlabel('β2')
# plt.ylabel('β1')
# clrbar = plt.colorbar(cntr, shrink=0.7)
# clrbar.set_label('RSS')
# plt.show()

#############################################################################################################