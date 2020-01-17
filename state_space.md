# Level-trend State Space Models (SSMs)
## A PyTorch implementation from scratch

A State Space Model (SSM) is a model that uses state variables to describe a system by a set of first-order differential equations.
They model the temporal structure of the data via a latent state l_t ∈ R that can be used to encode time series components such as level, trend, and seasonality patterns.
Among the infinite different flavors of SSMs that this definition potentially comprises I will focus on the simple linear-trend model, where the latent state l only references the level of a time series and its instantaneous trend.
State variables evolve over time in response to externally provided inputs, so that they can encode all the characteristics of a system and predict its behavior.
The main components of a SSM are a **transition** model, which describes how the latent state evolves over time and an observation model, which tells how to go from the latent space variables to the values we observe.

Let us start by defining the transition equation of our model:

![Pv_i](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20l_t%20%3D%20F_tl_%7Bt-1%7D%20&plus;%20g_t%20%5Cepsilon%20%5Ctextrm%7B%2C%7D%20%5Cquad%20%5Cepsilon%20%5Csim%20N%280%2C1%29)


