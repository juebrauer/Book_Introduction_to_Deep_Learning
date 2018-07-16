"""
File: som.py

Here we define a class that implements the
Self-Organizing Map (SOM) neural network model.

A SOM is an unsupervised learning algorithm that
allows to distribute N prototype vectors ("neurons")
automatically in the input space.

It can be used for dimensionality reduction and
clustering.
"""

import numpy as np

from som_neuron import som_neuron

class som:

    list_neurons = []
    nr_neurons = 0
    nr_steps_trained = 0
    BMU_nr = -1

    """
    Create a new Self-Organizing Map
    with the desired number of neurons.
    
    Each neuron will store input_dim many
    weights.
    """
    def __init__(self, input_dim, nr_neurons, nr_classes):

        # store number of neurons
        self.nr_neurons = nr_neurons

        # create the desired number of neurons
        for i in range(nr_neurons):

            # neurogenesis: create a new neuron
            neuron = som_neuron(input_dim, nr_classes)

            # store the new neuron in a list
            self.list_neurons.append( neuron )

        # prepare matrix for 2D neighborhood
        # topology
        S = int(np.sqrt(nr_neurons))
        self.neighborhood = \
            np.arange(nr_neurons).reshape(S,S)

        print("Neuron neighborhood:\n", self.neighborhood)


    """
    Initializes the neuron positions to
    the specified rectangle
    """
    def initialize_neuron_weights_to_grid(self, rectangle):

        S = int(np.sqrt(self.nr_neurons))

        orig_x = rectangle[0]
        orig_y = rectangle[1]
        width  = rectangle[2]
        height = rectangle[3]

        for id in range(self.nr_neurons):

            # get the next neuron
            neuron = self.list_neurons[id]

            # compute a 2D coordinate in input space
            # to initialize the weight vector with this
            # 2D coordinate
            grid_y = int(id / S)
            grid_x = id % S
            ispace_x = orig_x + grid_x * (width  / S)
            ispace_y = orig_y + grid_y * (height / S)

            # store that coordinates
            neuron.weight_vec[0] = ispace_x
            neuron.weight_vec[1] = ispace_y


    """
    Initializes the neuron positions to origin    
    """
    def initialize_neuron_weights_to_origin(self):

        for id in range(self.nr_neurons):

            # get the next neuron
            neuron = self.list_neurons[id]

            # store that coordinates
            neuron.weight_vec[0] = 0
            neuron.weight_vec[1] = 0


    """
    Returns all the neighbors of a given neuron
    Example:
    2D Neuron neighborhood of 49 neurons arranged in a 7x7 grid:
         [[ 0  1  2  3  4  5  6]
          [ 7  8  9 10 11 12 13]
          [14 15 16 17 18 19 20]
          [21 22 23 24 25 26 27]
          [28 29 30 31 32 33 34]
          [35 36 37 38 39 40 41]
          [42 43 44 45 46 47 48]]
    """
    def get_neighbors(self, id):

        N = self.nr_neurons
        S = int(np.sqrt(N))

        # case #1: corner?

        # top left corner
        if id==0:
            return [1,S]

        # top right corner
        if id==S-1:
            return [S-2,2*S-1]

        # bottom left corner:
        if id==N-S:
            return [N-S-S, N-S+1]

        # bottom right corner:
        if id==N-1:
            return [N-1-S, N-1-1]


        # case #2: border?
        y = int(id / S)
        x = id % S

        # top border
        if (y==0):
            return [id-1,id+1,id+S]

        # bottom border
        if (y==S-1):
            return [id-1,id+1,id-S]

        # left border
        if (x==0):
            return [id-S,id+S,id+1]

        # right border
        if (x==S-1):
            return [id-S,id+S,id-1]


        # case #3: normal cell?
        return [id-S,id-1,id+1,id+S]



    """
    Given an input vector:
    determine the Best Matching Unit (neuron)
    """
    def compute_bmu(self, input_vec):

        # 1. let all neurons comput their output values
        for neuron_nr in range(self.nr_neurons):
            # get the next neuron
            neuron = self.list_neurons[neuron_nr]

            # compute new output value of neuron
            neuron.compute_output(input_vec)

        # 2. now determine the Best Matching Unit (BMU),
        #    i.e., the neuron with the smallest output
        #    value (each neuron computes the distance of
        #    its weight vector to the input vector)
        self.BMU_nr = 0
        minimum_dist = self.list_neurons[0].output
        for neuron_nr in range(1, self.nr_neurons):

            if (self.list_neurons[neuron_nr].output < minimum_dist):
                minimum_dist = self.list_neurons[neuron_nr].output
                self.BMU_nr = neuron_nr



    """
    Train the SOM with one more training vector,
    i.e.,
    - determine the Best Matching Unit (BMU)
    - adapt the BMU and its neighbored neurons
      into the direction of the input vector      
    """
    def train(self, input_vec, learn_rate, adapt_neighbors, classinfo):

        self.nr_steps_trained += 1

        # 1. determine the BMU for this input vector
        self.compute_bmu(input_vec)
        #print("The BMU is neuron #", BMU_nr,
        #      "and the distance is", minimum_dist)


        # 2. now move the BMU a little bit into the direction
        #    of the input vector
        BMU_neuron = self.list_neurons[self.BMU_nr]
        BMU_neuron.adapt_to_vector(input_vec, learn_rate)
        BMU_neuron.inc_class_counter( classinfo )


        # 3. now get the list of all neighbors of the BMU
        #    and move the neighbors a little bit into the
        #    direction of the input vector as well
        if (adapt_neighbors):

            neighbors = self.get_neighbors(self.BMU_nr)

            # for all neighbors of this neuron:
            for j in neighbors:

                # get that neuron
                neighbored_neuron = self.list_neurons[j]

                # adapt the neighbored neuron to input vector
                # as well
                neighbored_neuron.adapt_to_vector(input_vec, learn_rate/2.0)

