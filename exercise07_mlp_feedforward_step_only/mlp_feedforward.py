import numpy as np
import math


def func_identity(x):
    return x

def func_sigmoid(x):
    return 1 / (1 + math.exp(-x))

func_identity = np.vectorize(func_identity)
func_sigmoid  = np.vectorize(func_sigmoid)

class TF:
    identity = 1
    sigmoid = 2



class mlp_feedforward:

    nr_layers = 0
    nr_neurons_per_layer = []
    tf_per_layer = []
    weight_matrices = []
    neuron_output_vecs = []

    def __init__(self):
        print("Generated a new empty MLP")


    def show_architecture(self):

        print("MLP architecture is now: ", end=" ")

        for i in range(self.nr_layers):
            print(str(self.nr_neurons_per_layer[i]), end=" ")

        print("\n")


    """
    Adds a new layer of neurons
    """
    def add_layer(self, nr_neurons, transfer_function):

        # 1. store number of neurons of this new layer
        #    and type of transfer function to use
        self.nr_neurons_per_layer.append( nr_neurons )
        self.tf_per_layer.append( transfer_function )

        # 2. generate a weight matrix?
        if self.nr_layers>=1:

            # 2.1 how many neurons are there in the
            #     previous layer?
            nr_neurons_before =\
                self.nr_neurons_per_layer[self.nr_layers-1]

            # 2.2 initialize weight matrix with random
            #     values from (0,1)
            #     Do not forget the BIAS input for each
            #     neuron! For this: nr_neurons_before + 1
            W = np.random.uniform(low=0.0, high=1.0,
                                  size=(nr_neurons_before+1,nr_neurons))

            # 2.3 store the new weight matrix
            self.weight_matrices.append(W)

            # 2.4 output some information about the
            #     weight matrix just generated
            print("Generated a new weight matrix W. Shape is",
                  W.shape)
            size = W.nbytes/1024.0
            print("Size of weight matrix in KB is {0:.2f}".format(size))
            # print("W=",W)


        # 3. generate a new neuron output vector
        out_vec = np.zeros(nr_neurons)
        self.neuron_output_vecs.append( out_vec )

        # 4. update number of layers
        self.nr_layers += 1

        # 5. show current MLP architecture
        self.show_architecture()


    """
    Given an input vector, we compute
    the output of all the neurons layer by layer
    into the direction of the output layer
    """
    def feedforward(self, input_vec):

        # 1. set output of neurons from first layer
        #    to input vector values
        N = len(input_vec)
        self.neuron_output_vecs[0] = input_vec

        # 2. now compute neuron outputs layer by layer
        for layer_nr in range(1,self.nr_layers):

            #print("Computing output values for neurons in "
            #      "layer", layer_nr)

            # 2.1 get output vector previously computed
            o = self.neuron_output_vecs[layer_nr-1]

            # 2.2 add bias input
            o = np.append([1], o)

            # 2.3 vectors are one-dimensional
            #     but for matrix*matrix multiplication we need
            #     a matrix in the following
            N = len(o)
            o_mat = o.reshape(1,N)

            # 2.4 now get the right weight matrix
            W = self.weight_matrices[layer_nr-1]

            # 2.5 compute the product of the output (vector)
            #     and the weight matrix to get the output values
            #     of neurons in the current layer
            #print("shapes: o_mat=",o_mat.shape, "W=", W.shape)
            o_mat_this_layer = np.matmul(o_mat,W)

            # 2.6 apply transfer function
            if self.tf_per_layer[layer_nr]==TF.sigmoid:
                o_mat_this_layer = func_sigmoid(o_mat_this_layer)
            else:
                o_mat_this_layer = func_identity(o_mat_this_layer)

            # 2.7 we want to store a vector and not a matrix
            self.neuron_output_vecs[layer_nr] = o_mat_this_layer.flatten()



    """
    Show output values of all neurons
    in the specified layer
    """
    def show_output(self, layer):

        print("output values of neuron in layer",layer,":",
              self.neuron_output_vecs[layer])