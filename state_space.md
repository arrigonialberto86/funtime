# Level-trend State Space Models (SSMs)
## A PyTorch implementation from scratch

A State Space Model (SSM) is a model that uses state variables to describe a system by a set of first-order differential equations.
They model the temporal structure of the data via a latent state l_t ∈ R that can be used to encode time series components such as level, trend, and seasonality patterns.
Among the infinite different flavors of SSMs that this definition potentially comprises I will focus on the simple linear-trend model, where the latent state l only references the level of a time series and its instantaneous trend.
State variables evolve over time in response to externally provided inputs, so that they can encode all the characteristics of a system and predict its behavior.
The main components of a SSM are a **transition** model, which describes how the latent state evolves over time and an observation model, which tells how to go from the latent space variables to the values we observe.

Let us start by defining the transition equation of our model:

![Pv_i](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20l_t%20%3D%20F_tl_%7Bt-1%7D%20&plus;%20g_t%20%5Cepsilon%20%5Ctextrm%7B%2C%7D%20%5Cquad%20%5Cepsilon%20%5Csim%20N%280%2C1%29)

Here you see a deterministic transition matrix F_t and a random innovation component summarized in vector ![ge](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20g_t%5Cepsilon)
As this is the general form of any SSM, we need to list the instantiation parameters of our simple level-trend model:

![param](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20F_t%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%201%5C%5C%200%20%26%201%20%5Cend%7Bbmatrix%7D%20%5Ctextrm%7B%2C%7D%5Cquad%20g_t%3D%5Cbegin%7Bbmatrix%7D%20%5Calpha%5C%5C%20%5Cbeta%20%5Cend%7Bbmatrix%7D)

The previous equation refers to the dynamics of evolution of the latent state, which in turn generats real observations z_t through an *observation* model:

![obs](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20z_t%20%3D%20y_t%20&plus;%20%5Csigma%20%5Cepsilon%20%5Ctextrm%7B%2C%7D%20%5Cquad%20y_t%20%3D%20a%5ETl_%7Bt%20-%201%7D%20&plus;%20b%20%5Ctextrm%7B%2C%7D%20%5Cquad%20%5Cepsilon%20%5Csim%20N%280%2C%201%29)

The parameters that specify this model are assumed to be fixed in time for our purposes, although in recent papers such as [this](https://papers.nips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting.pdf) the parameters are predicted by a RNN at each time point, effectively creating a time-varying Deep State Space Model.
The backpropagation is simply performed by perfoming Kalman filtering (more on that later) and calculating the negative log likelihood at each time point using the Gaussian distribution defined by the observation model above. 

Let us start our description by simulating some time series using the generative model reported above and PyTorch:

```python
import torch
import matplotlib.pyplot as plt
%matplotlib inline
fig= plt.figure(figsize=(12,6))

def next_obs(current_l=torch.Tensor([0, 0]), a=torch.Tensor([1, 1]), 
             F = torch.Tensor([[1, 1], [0, 1]]), alpha=0.6, beta=0.6, 
             sigma_t=torch.Tensor([3])): 
    g = torch.Tensor([alpha, beta])
    y_t = torch.matmul(a.T, current_l)
    z_t = y_t + sigma_t * torch.empty(1).normal_(mean=0,std=1)
    next_l = torch.matmul(F, current_l) + g * torch.empty(1).normal_(mean=0,std=1)
    return next_l, z_t

def simulate_single_ts(n_obs = 39):
    sim_list = []
    current_obs = torch.Tensor([0, 3])
    for i in range(n_obs):
        current_obs, z_t = next_obs(current_l=current_obs)
        sim_list.append(z_t)
    return sim_list

for i in range(39):
    plt.plot(simulate_single_ts())
plt.title('Random generation of time series using a linear state-space model')
```

<img src="state_space/random_ts.png" alt="Image not found" width="600"/>

I have talked above about the Kalman filter. Although it is not in the scope of this post a detailed explanation of the mechanics of filtering I will just go ahead and list the formulas that will be implemented in the below.
Kalman filtering is composed of three recurrent steps:

## Forecasting

![forecast](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20%5Cmu_t%20%3D%20a%5E%7BT%7Df_t%20%5Ctexrm%7B%2C%7D%5Cquad%20%5CSigma%3Da%5E%7BT%7DSa%20&plus;%20%5Csigma%5E2_1)

Then (to be clear) sample from the forecast distribution by assuming a Gaussian likelihood:

![sample](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20x_t%20%5Csim%20N%28%5Cmu_t%2C%20%5Csigma_t%29)

## Updating (state filtering)

![mean update](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20f_t%20%3D%20f_%7Bt-1%7D%20&plus;%20S_%7Bt-1%7Da%5CSigma%5E%7B-1%7D%28z_%7Bobs%7D%20-%20%5Cmu_%7Bt-1%7D%29)

![cov_update](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20S_t%20%3D%20S_%7Bt-1%7D%20-%20S_%7Bt-1%7Da%5CSigma%5E%7B-1%7Da%5E%7BT%7DS_%7Bt-1%7D)

## State prediction

![mu pred](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20f_%7Bt&plus;1%7D%20%3D%20Fx_%7Bt%7D)

![cov_pred](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20S_%7Bt&plus;1%7D%20%3D%20FS_tF%5ET%20&plus;%20gg%5ET)



## Equation editor
- https://www.codecogs.com/latex/eqneditor.php (Latin Modern, 12pts, 150 dpi)