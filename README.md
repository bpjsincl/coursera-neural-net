## Project Description

Assignments for Geoffrey Hinton's Neural Net Course on Coursera, translated from Matlab into Python.

* assignments 2-4 are quite different than what is presented in the course, as they were refactored into logical
classifiers (adapted from the sklearn framework).
* more work could certainly be done to remove redundancy between assignments, especially between 3 and 4.
* course can be found here: https://www.coursera.org/course/neuralnets

## Assignment 1
* Implements linear Perceptron for two class problem

## Assignment 2
* Implements a basic framework for training neural nets with mini-batch gradient descent for a language model.
* Assignment covers hyperparameter search and observations through average cross entropy error.
    * i.e. number of training epochs, embedding and hidden layer size, training momentum

## Assignment 3
* Trains a simple Feedforward Neural Network with Backpropogation, for recognizing USPS handwritten digits.
* Assignment looks into efficient optimization, and into effective regularization.
* Recognizes USPS handwritten digits.

## Assignment 4
* Trains a Feedforward neural network with pretraining using Restricted Boltzman Machines (RBMs)
* The RBM is used as the visible-to-hidden layer in a network exactly like the one made in programming assignment 3.
* The RBM is trained using Contrastive Divergence gradient estimator with 1 full Gibbs update, a.k.a. CD-1.
* Recognizes USPS handwritten digits.
