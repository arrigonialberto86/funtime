# Transfer learning: making your ConvNets great since 2014

### Convolutional neural networks (CNN): the context
Many deep neural networks trained on natural images exhibit a curious phenomenon in common: on the first layer they learn features similar to Gabor filters and color blobs. The appearance of these filters is very common, and made researchers understand that such first layers are not _specific_ for a certain task or a dataset, but they can truly build the foundation of a _general_ learning system.
This phenomenon occurs not only for different datasets, but even with very different training objectives, including supervised image classification (Krizhevsky et al., 2012), unsupervised density learning (Lee et al., 2009), and unsupervised learning of sparse  representations (Le et al., 2011).
On the other hand, generalization power is progressively lost by the terminal layers of the network, as neurons become more and more specialized and tied to the _specific_ classification problem they have been trained on.

<img src="transfer_learning/image_1.png" alt="Image not found" width="600"/>
======
__Fig.1__: Outer layers in convnets tend to learn features similar to Gabor filters and color blobs

### What is 'transfer learning'?
It is often difficult (and computationally very expensive) to reach the amount of data needed to train CNNs (which is in order of tens of thousands of images). Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet as a fixed feature extractor for the task of interest (Razavian et al., 2014)


This means: take a ConvNet that has been pre-trained on ImageNet (a huge repository of images collected for this purpose), remove the last fully-connected layer (which as we have shown is specific for the task the network was trained for), then treat the rest of the ConvNet as a feature extractor for the new dataset. Once you extract the features for all images, train a classifier for the new dataset.

The effectiveness of transfer learning is supported by a vast amount of evidence, and some recent (empirical) findings have spurred much interest in the methodology, as:
- the transferability of features decreases as the distance between the base task and target task increases, but transferring features even from distant tasks can be better than using random features (Yosinski et al., 2014)
- initializing a network with transferred features from almost any number of layers can produce a boost to generalization that lingers even after fine-tuning to the target dataset (Yosinski et al., 2014)

<img src="transfer_learning/image_2_bis.png" alt="Image not found" width="800"/>
======
__Fig.2__: Transfer learning: the green layers are borrowed from the first network and supplied as feature extractors to the second network (with frozen weights)


Transfer learning takes two slightly different approaches:
- __ConvNet as fixed feature extractor__ Take a ConvNet pretrained on ImageNet, remove the last fully-connected layer (this layer’s outputs are the 1000 class scores for a different task like ImageNet), then treat the rest of the ConvNet as a fixed feature extractor for the new dataset (as depicted in Fig.2)
- __Fine-tuning the ConvNet__ The second strategy is to not only replace and retrain the classifier on top of the ConvNet on the new dataset, but to also fine-tune the weights of the pretrained network by continuing the backpropagation. It is possible to fine-tune all the layers of the ConvNet, or it’s possible to keep some of the earlier layers fixed (due to overfitting concerns) and only fine-tune some higher-level portion of the network (more on this later)


