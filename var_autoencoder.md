# Variational autoencoder
## Variational Autoencoders (VAEs): view on deep learning 

### Autoencoders in brief 
An autoencoder consists of two parts, an encoder and a decoder. The encoder will read the input and compress it to a compact representation, and the decoder will read the compact representation and recreate the input from it.

One way to obtain useful features from the autoencoder is to constrain `h` to have smaller dimension than `x`. An autoencoder whose code dimension is less than the input dimension is called undercomplete. Learning an undercomplete representation forces the autoencoder to capture the most salient features of the training data.

Traditionally, autoencoders were used for dimensionality reduction or feature learning. Recently, theoretical connections between autoencoders and latent variable models have brought autoencoders to the forefront of generative modeling, and variational autoencoders have become one of the hottest topics around.

<img src="https://latex.codecogs.com/gif.latex?\dpi{200}&space;\fn_phv&space;\small&space;x^{y}&space;&plus;&space;18&space;/&space;4" title="\small x^{y} + 18 / 4" />

<img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\fn_phv&space;\small&space;x^{y}&space;&plus;&space;18&space;/&space;4" title="\small x^{y} + 18 / 4" />

### Applications of autoencoders
