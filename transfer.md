# Transfer learning: making your convnets great (since 2015)

Many deep neural networks trained on natural images exhibit a curious phenomenon in common: on the first layer they learn features similar to Gabor filters and color blobs. The appearance of these filters is very common, and made researchers understand that such first layers are not _specific_ for a certain task or a dataset, but they can truly build the foundation of a _general_ learning system.
This phenomenon occurs not only for different datasets, but even with very different training objectives, including supervised image classification (Krizhevsky et al., 2012), unsupervised density learning (Lee et al., 2009), and unsupervised learning of sparse  representations (Le et al., 2011).
On the other hand, generalization power is progressively lost by the last layers of the network, as neurons become more and more specialized and tied to the _specific_ classification problem they have been trained on.

The effectiveness of transfer learning is supported by a vast amount of evidence, and some recent (empirical) findings have 
- the transferability of features decreases as the distance between the base task and target task increases, but transferring features even from distant tasks can be better than using random features (Yosinski et al., 2014)
- initializing a network with transferred features from almost any number of layers can produce a boost to generalization that lingers even after fine-tuning to the target dataset (Yosinski et al., 2014)


