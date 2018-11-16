# "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" 
## Tensorflow implementation of the DeepMind paper presented at NIPS in late 2017

The thing we love about probabilistic programming is of course the possibility to model the uncertainty of model predictions. This convenient property comes with a cost unfortunately, as the calculations needed to perform e.g. Markov Chain Monte Carlo are very expensive.
In this short but insightful paper Lakshminarayanan et al., report a simple method to assess model uncertainty using NNs, which compares well even to Bayesian networks (as they demonstrate in the paper).
Let us start by listing the key highlights of this paper:
- The authors identify a scoring rule for uncertainty estimation in NNs
- They use ensembles of NNs + adversarial training examples and evaluate the predictions in different contexts.

## The scoring rule
Scoring rules assess the quality of probabilistic forecasts, by assigning a numerical score based on the predictive distribution and on the event or value that generates. ()You may want to check out this nice review [Gneiting et al., 2007] for more information on scoring functions, but it is not required to follow the rest of this post).
But let us concentrate on regression problems for a moment: in this context NNs output a single value ([mu](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Chuge%20%5Cmu%28x%29)) and the weights are optimized via backprop to minimize the Mean Squared Error (MSE) on the training set. Thus, we are not taking into consideration predictive uncertainty at all. In order to change this, we can propose a slight modification of the final layer, so that it produces two outputs (mean [mu](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Chuge%20%5Cmu%28x%29)) and variance [sigma](https://latex.codecogs.com/gif.latex?\dpi{150}&space;\LARGE&space;\sigma^2(x)).  


