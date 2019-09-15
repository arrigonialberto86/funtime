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
The following is the generative model for data points ![v_i](https://latex.codecogs.com/gif.latex?%5Cdpi%7B200%7D%20%5Clarge%20v_i):

![Pv_i](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cfn_phv%20P%28v_i%20%7C%20z_i%20%3D%20k%2C%20%5Cmu_k%29%20%5Csim%20N%28%5Cmu_k%2C%20%5Csigma%5E2%29)


## How to calculate the posterior (from book)

## The Chinese restaurant process 

    How to simulate the arrival process
    https://github.com/crcrpar/crp/blob/master/crp.py
    

## Inference on the number of clusters
    https://www.ritchievink.com/blog/2018/06/05/clustering-data-with-dirichlet-mixtures-in-edward-and-pymc3/
    does not work properly