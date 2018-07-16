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
    nr_classes = -1

    def __init__(self, nr_weights, nr_classes):

        self.weight_vec     = np.zeros(nr_weights)
        self.class_counters = np.zeros(nr_classes, dtype=np.int)
        self.nr_classes     = nr_classes


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


    """
    Each neuron (which corresponds to some feature in the world)
    also holds a class counter array.
    During learning we can record how often a certain neuron
    (feature) has been activated for which class.
    This will allow to vote for certain classes later
    if the neuron is activated and thereby allows for
    classification.
    """
    def inc_class_counter(self, for_which_class):

        self.class_counters[for_which_class] +=1


    def show_class_counters(self):

        for i in range(self.nr_classes):
            print(self.class_counters[i], end=" ")
        print()
