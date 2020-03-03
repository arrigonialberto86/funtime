# DeepSets (NIPS 2018)

This is a long overdue piece on a paper I read last year and found to be very interesting. 
It suggests a relatively simple way to structure deep nets in order to handle 'sets' instead of ordered 'list' of elements.
Despite this unimpressive description, it has important practical applications and surely its range of applicability goes far
beyond what I thought of when I read it for the first time.

Sets in data are everywhere, and the class of models I will describe in the present post are set to attain *permutation invariance*
of the inputs. An intuitive explanation of what a permutation invariant function is a function for which this relationship holds true:
![perm](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20f%28a%2C%20b%2C%20c%29%20%3D%20f%28b%2C%20c%2C%20a%29%20%3D%20f%28c%2C%20b%2C%20a%29)

In the **supervised** setting modeling sets is relevant when dealing with mappings to a label that should be invariant to inputs ordering
e.g. sets of objects in a scene, multi-agent reinforcement learning, estimation of population statistics...
In the **unsupervised** setting sets are relevant to tasks such as set/audience expansion, where given a set of objects that are similar to 
each other (e.g. set of words {lion, tiger, leopard}), our goal is to find new objects from a large pool of candidates such that 
the selected new objects are similar to the query (Zaheer et al., 2018).

Using the same notation of the DeepSets paper (Zaheer et al. 2018) a function operating on a input X is a valid set function 
(i.e. **invariant** to the permutation of instances in X) iff it can be decomposed in the form ![t](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20%5Crho%28%5Csum_%7Bx%20%5Cepsilon%20X%7D%20%5Cphi%28x%29%29)

The two "sub"-functions ![rho](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20%5Crho) and
![phi](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20%5Cphi) are parameterized by neural networks, and by this 
simple formula you can tell that there is a shared branch of the net (defined by ![phi](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20%5Cphi))
which processes set elements independently, which are later *pooled* by ![rho](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20%5Crho).
You can picture this process as well by regarding these two functions as `map` ops followed by `reduce`.

## The architecture

Let us now list the required architectural elements that compose a DeepSet neural network:
- Each input element of the set if transformed (possibily by several layers) of the 'mapping' function ![phi](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20%5Cphi)
- The resulting vectors are summed up together and the output is in turn transformed by using the ![rho](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20%5Crho)
network (the same as any other deep network).


 

## Reference
- https://www.inference.vc/deepsets-modeling-permutation-invariance/

## Equation editor
- https://www.codecogs.com/latex/eqneditor.php (Latin Modern, 12pts, 150 dpi)