# Dirichlet mixture processes
## Where do people sit in a Chinese restaurant?

## Introduction and intuition
I have been wanting to write about Dirichlet processes (DP) for some time now, but I have never had the chance to wrap my mind around this topic which I consider to be one the most challenging of modern statistics.
In particular, I found hard to understand how the famous Chinese restaurant process (CRP) is intimately linked to the abstract concept behind a Dirichlet process.
This was until I read Chapter X of this book, where the author shows how the CRP is a natural descendant (or literally the same thing) of a Dirichlet process.

From these, the author shows how DPs are intimately related to clustering, a class of algorithms for which the number of clusters is iteratively defined by the process itself. 
In the following paragraphs, I will try to give a concise explanation of DP processes and how they could be used to determine the number of clusters for a dataset. 

## Intuition and formal model description (https://en.wikipedia.org/wiki/Dirichlet_process, example 2)
A Dirichlet process is a distribution over distributions, so instead of generating a single parameter (vector) a single draw from a DP outputs another (discrete) distribution. 
As tricky as it sounds, we can develop some intuition for DP through the following example borrowed from genomics data analysis. Let's say we have a a dozen blood samples of the same person over the course of a treatment: 
for each blood sample we have measurements for about ~30 k genes and we are interested in capturing patterns of co-espressions, i.e. genes whose expression co-vary hinting at shared regulatory processes.
One obvious solution to this is clustering, although it be may hard (if not impossible) to decide a priori what the number of clusters will be (assuming we plan to use K-means).
The number of co-expression patterns is not known, moreover clustering itself is used as a tool to identify novel co-regulatory circuits that could be targeted in therapeutic settings. 
By using a Dirichlet process we circumvent the need to specify the number of clusters ahead of time.

Let's naively start modeling our dataset by supposing that there are K clusters of normally distributed expression patterns (as in microarray experiments) and that the variance `sigma` is known. 
I'll now use a notation which may be easier to understand for those already familiar with mixture models. 
The following is the generative model for data points ![v_i](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20v_i):

![Pv_i](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20P%28v_i%20%7C%20z_i%20%3D%20k%2C%20%5Cmu_k%29%20%5Csim%20N%28%5Cmu_k%2C%20%5Csigma%5E2%29)

And the probability that ![z_i](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20z_i) is equal to k is equal to ![pi](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20%5Cpi_k), which can be regarded
as a prior on the cluster k as with Gaussian mixture models:

![pz_i](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20P%28z_i%20%3D%20k%29%20%3D%20%5Cpi_k)

This prior is drawn from a Dirichlet distribution which is symmetric (we have no initial information distinguishing the clusters):

![pr](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20P%28%5Cpi%20/%20%5Calpha%29%20%5Csim%20Dir%28%5Cfrac%7B%5Calpha%7D%7BK%7D%20*%20%5Ctextbf%7B1%7D_K%29)

...knowing that every cluster mean ![mu_ki](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20%5Cmu_k) is drawn from a base distribution `H` (which for most applications can be a Gaussian distribution with 0 mean and standard deviation equal to 1):

![mu_k](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20%5Cmu_k%20%5Csim%20H%28%5Clambda%29)

What this all means is that we'll put identical priors ![hhh](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20H%28%5Clambda%29) over every ![mmm](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20%5Cmu_k) that is generated,
and that according to what we know they are all equally probable, as indicated by the parametrization of the Dirichlet prior distribution. 
The parameters ![alpha](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20%5Calpha) and ![lambda](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20%5Clambda) are instead fixed parameters, reflecting
our prior knowledge of the system we are modeling.

What written in the paragraph above refers to a Dirichlet mixture model of K component, i.e. we must know the number of clusters in advance in order to assign observations to clusters.
In order to understand what a DP is though, we need to rewrite our model using a slightly different notation:

![v_inew](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20P%28v_i%20%7C%20%5Ctilde%5Cmu_i%29%20%5Csim%20N%28%5Ctilde%5Cmu_i%2C%20%5Csigma%5E2%29)

Let's stop to think for a second about the subscript `i`: we are not reasoning anymore in terms of clusters `k`, we suppose instead that every observation `i` in our dataset is associated with a parameter
![mu_i_tilde](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20%5Ctilde%5Cmu_i), which is in turn drawn from some discrete distribution G with support on the K means:

![G](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20%5Ctilde%5Cmu_i%20%5Csim%20G%20%3D%20%5Csum_%7Bk%3D1%7D%5E%7BK%7D%20%5Cpi_k%20%5Cdelta_%7B%5Cmu_k%7D%28%5Ctilde%5Cmu_i%29)


Where ![deltamu](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20%5Cdelta_%7B%5Cmu_k%7D) is the Dirac delta, which is basically an indicator function centered on ![indi](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20%5Cmu_k)
And again, we draw ![pi](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20%5Cpi) from a symmetric Dirichlet distribution:

![pr](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20P%28%5Cpi%20/%20%5Calpha%29%20%5Csim%20Dir%28%5Cfrac%7B%5Calpha%7D%7BK%7D%20*%20%5Ctextbf%7B1%7D_K%29)

...and we sample ![mu_k](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20%5Cmu_k) as usual:

![mu_k](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20%5Cmu_k%20%5Csim%20H%28%5Clambda%29)

We now have all the tools to describe a Dirichlet process, just by extending the sum in the G distribution definition to an infinite number of clusters K: ![infinity](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20G%28%5Ctilde%5Cmu_i%29%20%3D%20%5Csum_%7Bk%3D1%7D%5E%7B%5Cinfty%7D%20%5Cpi_k%20%5Cdelta_%7B%5Cmu_k%7D%28%5Ctilde%5Cmu_i%29)
And we represent the overall model in a more compact way:

![v_inew](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20P%28v_i%20%7C%20%5Ctilde%5Cmu_i%29%20%5Csim%20N%28%5Ctilde%5Cmu_i%2C%20%5Csigma%5E2%29)

![mui](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20%5Ctilde%5Cmu_i%20%5Csim%20G)

Now remember that a Dirichlet process is defined as a distribution over distributions Gs, so we can write:

![G_draw](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20G%20%5Csim%20DP%28H%28%5Clambda%29%2C%20%5Calpha%29)

Now, a discrete prior distribution with an infinite number of components may constitute an interesting mind experiment, but of course we need to find a way to sample from this distribution, 
and moreover given a dataset `D` we would like to derive the posterior distribution ![post](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20P%28%5Cpi%2C%20%5Cmu%20%7C%20D%29).
Let's start by 

## The stick-breaking process: how to sample from a DP 

As we have seen in the previous paragraph, draws from a Dirichlet process are distributions over a set S


# // add part of code that simulate random draws from a DP process

## How to calculate the posterior (from book)

## The Chinese restaurant process 

    How to simulate the arrival process
    https://github.com/crcrpar/crp/blob/master/crp.py
    

## Inference on the number of clusters
    https://www.ritchievink.com/blog/2018/06/05/clustering-data-with-dirichlet-mixtures-in-edward-and-pymc3/

    
    
    
## Equation editor
- https://www.codecogs.com/latex/eqneditor.php (char is Helvetica, 10pts, 150 dpi)