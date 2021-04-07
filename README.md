## What is PyTorch?

PyTorch is a framework for building and training neural networks.
PyTorch has powerful modules that provides an efficient and more convenient way to build large neural networks, for example: nn module.


## What are Tensors? 

Tensors are the fundamental data structure in PyTorch or any other deep learning frameworks.

Tensors are just Numpy Arrays when them converted into vectors and matrices. PyTorch takes these tensors and makes it simple to move them to GPUs for the faster processing needed when training neural networks.
In general, PyTorch tensors can be added, multiplied, subtracted, etc, just like Numpy arrays. 

Neural network computations are just a bunch of linear algebra operations on tensors or matrices.


## What are Neural Networks?

Neural Networks are built from individual parts approximating neurons, typically called units or simply neurons. Each unit has some number of weighted inputs. These weighted inputs are summed together (a linear combination) then passed through an activation function to get the unit's output.


## What is Deep learning?

Deep learning neural networks tend to be massive with dozens or hundreds of layers, that's where the term "deep" comes from.


## What are fully-connected neural networks?

Fully-connected or dense networks mean that each unit in one layer is connected to each unit in the next layer. 

In fully-connected networks, the input to each layer must be a one-dimensional vector (which can be stacked into a 2D tensor as a batch of multiple examples). However, our images are 28x28 2D tensors "row vector", so we need to convert them into 1D vectors. Thinking about sizes, we need to convert the batch of images with shape (64, 1, 28, 28) to have a shape of (64, 784), 784 is 28 times 28. This is typically called flattening, we flattened the 2D images into 1D vectors.


## What is MNIST?

MNIST is a dataset that consists of images of greyscale handwritten digits. Each image is 28x28 pixels 2D.


## What is Fashion-MNIST?

Fashion-MNIST is a dataset of 28x28 greyscale images of clothes. It's more complex than MNIST.


## What is ImageNet?

ImageNet is a massive dataset with over 1 million labeled images in 1000 categories. It's used to train deep neural networks using an architecture called convolutional layers.


## What is Transfer learning?

Transfer learning is an optimization approach in deep learning (and machine learning) where knowledge is transferred from one model to another.

Using transfer learning, we can solve a particular task using full or part of an already pre-trained model in a different task to save time or get better performance.

****************************************************************************************************************

## This repo contains my reimplementation of [Udacity's Deep Learning with PyTorch](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-to-pytorch) lessons:

* lesson 1: Introduction to PyTorch and using tensors
* lesson 2: Building fully-connected neural networks with PyTorch
* lesson 3: How to train a fully-connected network with backpropagation on MNIST
* lesson 4: Exercise - train a neural network on Fashion-MNIST
* lesson 5: Using a trained network for making predictions and validating networks
* lesson 6: How to save and load trained models
* lesson 7: Load image data with torchvision, also data augmentation
* lesson 8: Use transfer learning to train a state-of-the-art image classifier for dogs and cats
