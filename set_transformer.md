# Paper review & (TensorFlow) code: Set Transformer
## Set Transformer, ICML 2019


In a [previous post](https://arrigonialberto86.github.io/funtime/deepsets.html) I talked about functions preserving 
*permutation invariance* with respect to input data (i.e. sets of data points such as items in a shopping cart).
If you are not familiar with this concept please review that post or refer to the original [DeepSet paper](https://arxiv.org/abs/1703.06114).

I this post we will go through the details of a recent paper that leverage the idea of attention and the overall transformer architecture
to attain input permutation invariance, and solves some of the shortcomings connected with the naive pooling operation we used in
DeepSets. You can find the original publication by Lee et al. [here](https://arxiv.org/pdf/1810.00825.pdf): Set Transformer: A Framework for Attention-based
Permutation-Invariant Neural Networks.


In the DeepSet publication the authors suggest to use a simple pooling function (*sum* or *mean*) to combine together different 
branches of a neural network (each one processing one data point independently but with shared weights). Although this simple solution
effectively works as intended, by pooling different vectors together we are 'squashing' the information contained in source data and 
losing information about higher-order interactions which may exist among members of the set.
The **Set Tranformer** architecture suggested in the aforementioned publication tackles these shortcomings by providing a richer
representation of input data that capture higher-order interactions and parametrizes the *pooling* operation so that no information
is lost after combining data points. 


## Paper contributions

It has been observed before [before](https://arxiv.org/abs/1706.03762) that the transformer architecture ([here](http://jalammar.github.io/illustrated-transformer/) 
you can find a very nice 'visual' explanation) without positional encoding
does not retain information regarding the order by which the input words are being supplied. Thus, it makes sense to suppose that
the attention mechanism (which constitutes the basis of the Transformer architecture) could be used to process sets of elements in the same way
as we have seen for the DeepSet architecture. In Lemma 1 of the their paper, Lee et al. demonstrate that the mean operator (i.e. a pooling function we have talked about)
is a special case of dot-product attention with softmax (i.e. the self-attention mechanism). 
If this sounds strange to you (or, on the other hand, too simple to be true) I can provide an informal explanation of what self-attention is,
and why it matters in so many different contexts. 

Let us say we have a set of `f`-dimensional elements/vectors (in the classical NLP context these would be one-hot-encoded vectors to be subsequently
projected down to an embedding space, here just feature vectors). Since here we are processing *sets* we do not care about the order, 
just know that we have a `n x f` input matrix `N`. We project this matrix into 3 separate new matrices that are referred to as **keys** (K), **queries** (Q), and **values** (V).
The projection matrices will be learned during training just like any other parameters in the model. During training the projection matrices will learn how to produce queries, 
keys and values that will lead to optimal attention weights.

Let us just take a pair of feature vectors to make this concrete (and hopefully clearer): vector `a` and `b` both defined in ![R10](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20%7B%5CBbb%20R%7D%5E%7B10%7D).
We now use the three projection matrices K, Q and V (which are all trainable as said before) to obtain three versions (possibly very different according to parametrization)
of the input vector `a`, i.e. `a_Q`, `a_K`, `a_V`, whose dimensionality depends on K, Q, and V (this is an hyperparameter that needs tuning).
We do the same thing for `b` to obtain `b_Q`, `b_K`, `b_V`. The two-vector set (`{a, b}`) has now been converted to two lists of vectors.
What we would like to understand with these operations (the whole attention mechanism) is how much `a` is *related* to itself and `b`, or put in other terms:
do I need `b` when predicting a target related to `a` or all the information I need is already present in `a`?

We can calculate how related `a` is to itself first by multiplying (via the inner product) its query (a_q) and key (a_k) together. 
Remember, we compute all pairwise interactions between nodes include self-interactions, and unsurprisingly objects are likely to be related to themselves, 
but not necessarily since the corresponding queries and keys (after projection) may be different.

What we get by the queries (Q) and keys (K) product is a unnormalized attention weight matrix, which we later normalize by using a **softmax** function.
Once we have the normalized attention weights, we then multiply these by the corresponding value matrix (V) to focus on only certain salient parts of the value (V) matrix, and this will give us a new and updated node matrix:

![attention](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Clarge%20%5Chat%7BN%7D%20%3D%20softmax%28QK%5E%7BT%7D%29V)

The `QK^T` matrix is a `nxn` matrix which encoded the pairwise relationships between the elements of the input set.
We then multiply this by the value (V) matrix, which will update each feature vector according to its interactions with other elements, such that the final result is an updated set matrix.

What is truly remarkable about this encoding process is that is flexible enough to be useful in very different contexts: in NLP (where attention was conceived) 
the weight matrix represents how much one word is relevant for the translation of another word, i.e. the contextual information, in graph neural network the `QK^T` matrix
becomes the weighted adjacency matrix of the graph. 

## The network structure



## Reference
- "Attention is all you need", 2017, https://arxiv.org/abs/1706.03762
- "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks", 2019, https://arxiv.org/pdf/1810.00825.pdf

## Equation editor
- https://www.codecogs.com/latex/eqneditor.php (Latin Modern, 12pts, 150 dpi)
- {\Bbb R}^{10}