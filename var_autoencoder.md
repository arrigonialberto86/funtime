# Variational autoencoder
## The deep learning take on Variational Autoencoders (VAEs):

### Autoencoders in brief 
An autoencoder is a neural network that consists of two parts, an encoder and a decoder. The encoder reads the input and compresses it to a compact representation (stored in the hidden layer _h_), while the decoder reads the compact representation and recreates the input from it.

**Fig.1**: The general structure of an autoencoder, mapping an input x to an output (called reconstruction) r through an internal representation or code h. The autoencoder has two components: the encoder _f_ (mapping _x_ to _h_) and the decoder _g_ (mapping _h_ to
_r_) ('Deep learning book', Goodfellow et al., 2017)

One way to obtain useful features from the autoencoder is to constrain _h_ to have smaller dimension than _x_. An autoencoder whose code dimension is less than the input dimension is called 'undercomplete'. Learning an undercomplete representation forces the autoencoder to capture the most salient features of the training data, similarly to what other dimensionality reduction techniques do (e.g. PCA and multidimensional scaling).

For this reason, autoencoders were mainly used for dimensionality reduction or feature learning. Recently, theoretical connections between autoencoders and latent variable models have brought autoencoders to the forefront of generative modeling, and variational autoencoders have become one of the hottest topics in unsupervised learning.

### Applications of autoencoders



