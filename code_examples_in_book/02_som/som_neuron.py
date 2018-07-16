"""
File: som_neuron.py

A SOM neuron is a special type of neuron.
It stores a prototype vector in its weights.
Given an input vector, it computes some "distance"
between its input vector and its stored prototype vector.
"""

import numpy as np

class som_neuron:

    output = 0.0

    def __init__(self, nr_weights):

        self.weight_vec = np.zeros(nr_weights)


    def compute_output(self, input_vec):

        # this will compute the Euclidean distance between
        # the weight vector and the input vector
        self.output = np.linalg.norm(self.weight_vec-input_vec)


    """
    This will adapt the neuron's weight vector
    'a little bit' into the direction of the
    specified <adapt_vec>.
    """
    def adapt_to_vector(self, adapt_vec, learn_rate):

        delta_w = learn_rate * (adapt_vec - self.weight_vec)
        self.weight_vec += delta_w

