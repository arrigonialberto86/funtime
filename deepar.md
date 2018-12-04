# Paper review & code: Amazon DeepAR
## DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks

This blog post is about the DeepAR tool, which has been released by Amazon last summer, and integrated into SageMaker. 
Specifically, we will not use Amazon's implementation (which is presented in SageMaker as a black box), but I will try to present the theory and the challenges behind it, which are summarized in the paper mentioned in the title.

Demand forecasting is challenging, and it is especially difficult to handle at scale. 
I would like to briefly summarize here the critical points I have been facing in my day to day job:
- The "scale" problem: "*a lot*" of time series (in case you have been working at a company that has hundreds of product releases every year)
- A "grouped/clustered" structure of released products: categorical variables defining products' characteristics (e.g. product category, sales channel) 
- Cold-start products: products for which we do not have an historical time series (but we may have product category and other types of "shared" characteristics)
- Relevant co-variates: e.g. the weather, time of the year encodings etc...

There are many strategies to solve these problems (some of them are mentioned in the paper, 
such as matrix factorization methods [Yu et al.] and Bayesian approaches with hierarchical priors [Chapados et al.]), but none of them is easily scalable and handles *all* the problems listed above.

##The model##
What the authors suggest instead of fitting separate models for each time series is to create a *global* model from related time series to handle widely-varying scales through rescaling and velocity-based sampling.
They use an RNN architecture which incorporates a Gaussian/Negative Binomial likelihood to produce probabilistic forecasting and outperforms traditional single-item forecasting (the authors demonstrate this on several real-world data sets).
The figure below reports the architecture they use for training/prediction:

<img src="deepar/deepar_arch.png" alt="Image not found" width="800" />

Fig. 1 (left):
The idea behind this architecture is straightforward: the goal here is to predict at each time step the following (horizon=1). 
this means that the network must receive in input the previous observation (at lag=1) z_t-1, along with a set of (optional covariates x_i). The information is propagated to the hidden layer (represented in figure 1 by h)
and up to the likelihood function (which is a score function used here *in lieau* of a loss function). The likelihood function can be Gaussian of Negative Binomial, but
I will talk more on this later. As you can see in fig.1, during training (the network on the left) the error is calculated using the current parametrization of the likelihood *theta*.
Easily enough, this represent mu and sigma in the case of a Gaussian likelihood. This means that while performing backprop we are tuning the network
parameters (weights w) which change the parametrization of every e.g. Gaussian likelihood, until we converge to optimal values.

Fig 2 (right):
Once we have derived the parameters theta (e.g mu, sigma) for every time step during training (along with all the other weights of the network) it is time 
to predict the future: just remember that the prediction we have at each time step are distributions not single values.
We start by drawing one sample from the output distribution of the first time step: that sample is the input to the second time step and 
so on. Every time we start from the beginning and sample up to the prediction horizon we create the equivalent of Monte Carlo trace, which
means that in the end can calculate e.g. quantiles of the output distribution or assess uncertainty of the predictions.

We now turn to the likelihood model, which can be both Gaussian (with parametrization mu and sigma):

<img src="deepar/gaussian.png" alt="Image not found" width="800" />

or negative binomial when dealing with counts data. In case you have never used this kind of model, just think of it as an extension of a Poisson GLM
where we need to model the variance too (in Poisson models the mean is assumed to be equal to the variance, although when this is not the case we need some extra help to model "overdispersion"):

<img src="deepar/negative_binomial.png" alt="Image not found" width="800" />

The bottom line here is that the network is estimating the parameters through a custom layer which returns the likelihood parameters:

##About the covariates (feature)##
Features (x_i in paper and fig.1 notation) can be used to provide additional information about the item or the time point (e.g. week of year) to the model
They can also be used to include covariates that one expects to influence the outcome (e.g. price or promotion
status in the demand forecasting setting), as long as the features’ values are available also in the
prediction range.

##...and now some code##
Let us turn now 



