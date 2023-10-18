#!/usr/bin/env python
# coding: utf-8

# ### Part 1

# In[1]:


#pip install imageio --upgrade
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as imageio
from utils import *


# In[2]:


# read and display the original image
clean_img = imageio.imread('pic1.png').astype('float32')
plt.imshow(clean_img, cmap='gray')
plt.title('Original Image');


# In[3]:


# add a white Gaussian noise with standard deviation of sigma
sigma = 20
noisy_img = clean_img + np.random.normal(scale=sigma, size=clean_img.shape)
plt.figure()
plt.imshow(noisy_img, cmap='gray')
plt.title('Noisy Image');


# In[4]:


p = 8    # patch size
N = noisy_img.shape[0]
Y = extract_patches(noisy_img, patch_size=p)
assert Y.shape == (p**2, (N-p+1)**2), 'Invlid patch extraction'


# In[5]:


# implement MP and OMP sparse coding algorithms
# for each column of the dictionary, stop where norm(residual) ** 2 < D.shape[0] * tol ** 2

def MP(D: np.ndarray, Y: np.ndarray, tol:float) -> np.ndarray:
r = np.linalg.norm(Y[:,i])
    for i in range len(Y[0]):
        for j in range len(Y):
            e0[j] = 0
        e0[j] =  ( np.linalg.norm(Y[:,i])**2 - (np.dot(np.transpose(D[j,:]),Y[:,i]))**2 ) / np.dot(np.transpose(D[j,:]),D[j,:])
        if (e0[j]>e0[j-1]):
            c = e0[j]
            f = j
            r = r - c * D[f,:]
            
    raise NotImplementedError

# optional
def OMP(D: np.ndarray, Y: np.ndarray, tol:float) -> np.ndarray:
    raise NotImplementedError


# In[12]:


# implement MOD and KSVD dictionary update algorithms
# `DO NOT forget` to `normalize` columns of the dictionary

def MOD(Y:np.ndarray, X:np.ndarray):
    raise NotImplementedError


# In[14]:


C = 1.075    # noise gain
multiplier = 0.5    # fraction of the noisy image added to the final result

initial_dict = np.load('initial_dictionary.npy') # initialize dictionary to an overcomplete DCT dictionary
    
# visualize the initial dictionary
plt.figure(figsize=(3, 3))
plt.imshow(visualize_dict(initial_dict), cmap='gray');
plt.title('Initial Dictionary');


# ### MOD + MP

# In[ ]:


# training step
final_dict, X = train(Y, initial_dict, num_iter=10, C=1.075, sigma=sigma, sparse_code=MP, dict_update=MOD)


# In[ ]:


# visualize the final dictionary
plt.figure(figsize=(3, 3))
plt.imshow(visualize_dict(final_dict), cmap='gray');
plt.title('Final Dictionary');


# In[ ]:


# reconstruct the image and compare with the noisy and original one
denoised_img = reconstruct_image(initial_dict, X, noisy_img, p, m=0.1)
evaluate(clean_img, noisy_img, denoised_img)


# ### MOD + OMP

# In[ ]:


# training step
final_dict, X = train(Y, initial_dict, num_iter=10, C=1.075, sigma=sigma, sparse_code=OMP, dict_update=MOD)


# In[ ]:


# visualize the final dictionary
plt.figure(figsize=(3, 3))
plt.imshow(visualize_dict(final_dict), cmap='gray');
plt.title('Final Dictionary');


# In[ ]:


# reconstruct the image and compare with the noisy and original one
denoised_img = reconstruct_image(initial_dict, X, noisy_img, p, m=0.1)
evaluate(clean_img, noisy_img, denoised_img)


# ## Part 2
# 
# * Fit a polynomial of degree $K$ to all columns of the following dataset
# to predict number of "Cars" and "Buses" in the years of $2016$,$2017$,$2018$,$2019$ and $2020$.
# 
# 
# $$
# y_i = \sum_{k=0}^K w_k x_{i}^k + \epsilon_i
# $$
# 
# 
# Change the value of $K$ from 3 to 5. For each degree, plot the data and your model fit and compare the results.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as plt

dataset = pd.read_csv(u'data/arac.csv',sep=';')
dataset


# In[ ]:


BaseYear = 1966
K = ...


# In[ ]:


x = np.matrix(dataset.Year[0:]).T - BaseYear
cars = np.matrix(dataset.Car[0:]).T
buses = np.matrix(dataset.Bus[0:]).T


# In[ ]:


# In this part, change the value of K and show the results for "Cars"
# You may add new cells to show the results separately

# Create Vandermonde Matrix
A = np.hstack(np.power(x, i) for i in range(K + 1))

# Find the W matrix
# You may use linalg.lstsq here
...
f = ...


# Prediction
TargetYears = np.matrix([2016, 2017, 2018, 2019, 2020]).T
A2 = np.hstack(np.power(TargetYears, i) for i in range(K + 1))
# Predict the number of cars for target years
f2 = ...


# Plot the results
plt.plot(x + BaseYear, cars, 'o-')
plt.plot(x + BaseYear, f, 'r')
plt.plot(TargetYears, f2, 'ro-')

print("Predicted Numbers for cars")
print(f2)


# In[ ]:


# In this part, change the value of K and show the results for "Buses"
# You may add new cells to show the results separately

# Create Vandermonde Matrix
A = np.hstack(np.power(x, i) for i in range(K + 1))

# Find the W matrix
# You may use linalg.lstsq here
...
f = ...


# Prediction
TargetYears = np.matrix([2016, 2017, 2018, 2019, 2020]).T
A2 = np.hstack(np.power(TargetYears, i) for i in range(K + 1))
# Predict the number of cars for target years
f2 = ...


# Plot the results
plt.plot(x + BaseYear, buses, 'o-')
plt.plot(x + BaseYear, f, 'r')
plt.plot(TargetYears, f2, 'ro-')

print("Predicted Numbers for buses")
print(f2)

