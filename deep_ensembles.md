# Paper review & code: Deep Ensembles (NIPS 2017)
## Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles 
### Tensorflow implementation of the DeepMind paper presented at NIPS in late 2017

The thing we love about probabilistic programming is of course the possibility to model the uncertainty of model predictions. This convenient property comes with a cost unfortunately, as the calculations needed to perform e.g. Markov Chain Monte Carlo are very expensive.
In this short but insightful paper Lakshminarayanan et al., report a simple method to assess model uncertainty using NNs, which compares well even to Bayesian networks (as they demonstrate in the paper).
Let us start by listing the key highlights of this paper:
- The authors identify a scoring rule for uncertainty estimation in NNs
- They use ensembles of NNs + adversarial training examples and evaluate the predictions in different contexts.

## The scoring rule
Scoring rules assess the quality of probabilistic forecasts, by assigning a numerical score based on the predictive distribution and on the event or value that generates. ()You may want to check out this nice review [Gneiting et al., 2007] for more information on scoring functions, but it is not required to follow the rest of this post).
But let us concentrate on regression problems for a moment: in this context NNs output a single value (![mu](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cmu%28x%29)) and the weights are optimized via backprop to minimize the Mean Squared Error (MSE) on the training set. Thus, we are not taking into consideration predictive uncertainty at all. In order to change this, we can propose a slight modification of the final layer, so that it produces two outputs (mean ![mu](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cmu%28x%29)) and variance ![sigma](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csigma%5E2%28x%29) (which of course must be grater than 0).
We now minimize the negative log-likelihood criterion by treating the observed value as a sample from a Gaussian distribution with the predicted mean and variance:

![gausssian_likelihood](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20-logp_%5Ctheta%28y_n/x_n%29%20%3D%20%5Cfrac%7Blog%20%5Csigma%5E2%28x%29%7D%7B2%7D&plus;%5Cfrac%7B%28y-%5Cmu_%5Ctheta%28x%29%29%5E2%7D%7B2%5Csigma%5E2%28x%29%29%7D)

Thus, we need to implement the 1) custom loss function along with a 2) custom Keras layer that outputs both mu and sigma.
I'll take a step back to generate some data we can use to observe the properties of the model and what it really means to predict uncertainty in this context. Let us use the same function (y=x^3) referenced in the paper:

```python
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("white")


test_ratio = 0.1

data_range = 3
data_step = 0.005
data_sigma1 = 2
data_sigma2 = 1
num_data = 1

def pow_fun(x):
    return np.power(x, 3)

data_x = np.arange(-data_range, data_range + data_step, data_step)
data_x = np.reshape(data_x, [data_x.shape[0], 1])

data_y = np.zeros([data_x.shape[0], 1])
data_y_true = np.zeros([data_x.shape[0], 1])

for i in range(data_x.shape[0]):

    if (data_x[i,0] < 0): 
        data_y[i, 0] = pow_fun(data_x[i,0]) + np.random.normal(0, data_sigma1)
    else:
        data_y[i, 0] = pow_fun(data_x[i,0]) + np.random.normal(0, data_sigma2)
        
    data_y_true[i, 0] = pow_fun(data_x[i,0])
    
num_train_data = int(data_x.shape[0] * (1 - test_ratio))
num_test_data  = data_x.shape[0] - num_train_data

train_x = data_x[:num_train_data, :]
train_y = data_y[:num_train_data, :]
test_x  = data_x[num_train_data:, :]
test_y  = data_y[num_train_data:, :]

plt.rcParams['figure.figsize'] = [15, 10]
plt.axvline(x=0, linewidth=3)
plt.plot(data_x, data_y, '.', markersize=12, color='#F39C12')
plt.plot(data_x, data_y_true, 'r', linewidth=3)
plt.legend(['Data', 'y=x^3'], loc = 'best')

# plt.title('y = x^3 where $\epsilon$ ~ N(0, 3^2) and N(0, 1^2)')
plt.show()

```

<img src="deep_ensembles/first.png" alt="Image not found" width="600" />
 
The blue vertical line represents the point on the x-axis (0) where we have increased the dispersion of the generated data. We will try to understand if a model like the one presented here can adequately model the change in dispersion and vary the output sigma accordingly (on the training set).
At this point we need to generate the loss function and the custom layer:
```python
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Layer, Dropout
from keras.models import Model
from keras.initializers import glorot_normal
import numpy as np

def custom_loss(sigma):
    def gaussian_loss(y_true, y_pred):
        return tf.reduce_mean(0.5*tf.log(sigma) + 0.5*tf.div(tf.square(y_true - y_pred), sigma)) + 1e-6
    return gaussian_loss

class GaussianLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(GaussianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel_1 = self.add_weight(name='kernel_1', 
                                      shape=(30, self.output_dim),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.kernel_2 = self.add_weight(name='kernel_2', 
                                      shape=(30, self.output_dim),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.bias_1 = self.add_weight(name='bias_1',
                                    shape=(self.output_dim, ),
                                    initializer=glorot_normal(),
                                    trainable=True)
        self.bias_2 = self.add_weight(name='bias_2',
                                    shape=(self.output_dim, ),
                                    initializer=glorot_normal(),
                                    trainable=True)
        super(GaussianLayer, self).build(input_shape) 

    def call(self, x):
        output_mu  = K.dot(x, self.kernel_1) + self.bias_1
        output_sig = K.dot(x, self.kernel_2) + self.bias_2
        output_sig_pos = K.log(1 + K.exp(output_sig)) + 1e-06  
        return [output_mu, output_sig_pos]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.output_dim), (input_shape[0], self.output_dim)]

```

The implementation of the custom loss function is straightforward (although with a twist!): we need to encapsulate the loss function `gaussian_loss` into another function in order to pass the second parameter it needs to computer the log-likelihood (`sigma`).
Then, we can subclass Keras' `Layer` to produce our custom layer. There is an extensive documentation on this, see [Keras documentation](https://keras.io/layers/writing-your-own-keras-layers/), so I will just skip the details and just say that we basically need to implement three methods: `build`, `call`, `compute_output_shape`.
