# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% Imports
import numpy as np , random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#%% Functions
def linear_kernel(x, y):    
    return np.dot(x, y)

def objective(alpha):
     
#    x = [[ alpha[i] * alpha[j] * P[i][j] for i in range(N)] for j in range(N)]
    x = 0.5 * np.sum(np.dot(alpha,P)) - np.sum(alpha)
    return x
#    alpha_i_sum = np.sum(a)
def zerofun(x):
    return 0

def get_data(N):
    np.random.seed(100)
    classA = np.concatenate((np.random.randn(N, 2) * 0.2 + [1.5, 0.5],
                                   np.random.randn(N, 2) * 0.2 + [-1.5, 0.5])) 
    classB = np.random.randn(N*2, 2) * 0.2 + [0.0 , -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))
    N = inputs.shape[0] # Number of rows (samples)
    permute = list(range(N)) 
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    return inputs, targets
    
def plot(classA, classB):
    plt.plot([p[0] for p in classA],
             [p[1] for p in classA],
             'b.')
    plt.plot([p[0] for p in classB],
             [p[1] for p in classB],
             'r.')
    plt.axis('equal') #Force same scale on both axes
    plt.savefig('svmplot.pdf') #Save a copy in file
    plt.show() #Show the plot on screen
    
def get_p(inputs, target, N):
#    N = len(data)
    P = np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, N):
            P[i][j] = (target[i]) * (target[j]) * linear_kernel([(inputs[i])[0], (inputs[i])[1]], [(inputs[j])[0], (inputs[j])[1]])
    return P
#%% Script
N = 100
start = np.zeros(N)
#B = [(0, C) for b in range(N)] # upper bound
inputs, target = get_data(N)
B = [(0, None) for b in range(N)] # no upper bound
constraint = {'type': 'eq', 'fun': zerofun}
P = get_p(inputs, target, N)
ret = minimize(objective, start, bounds = B)#, constraints = constraint)
alpha = ret['x']
#plot(classA, classB)
#data = np.concatenate((inputs,target))
threshold = 0.000001
support = []
for i in range(len(alpha)):
    if alpha[i] > 10e-5:
        support.append((inputs[i][0], inputs[i][1], target[i], alpha[i]))
        

t = 0
K = 0
    
    
    
    
    
    
    
    
    
