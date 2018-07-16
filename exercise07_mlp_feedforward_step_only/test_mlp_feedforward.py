import numpy as np
from mlp_feedforward import mlp_feedforward
from mlp_feedforward import TF
from timeit import default_timer as timer

# 1. create a new MLP
my_mlp = mlp_feedforward()

# 2. which test to perform?
test = 0
if test == 0:

    # test MLP class with small network
    my_mlp.add_layer(5, TF.identity)
    my_mlp.add_layer(10,TF.identity)
    my_mlp.add_layer(5, TF.identity)
    my_mlp.add_layer(2, TF.identity)

    # overwrite random weights with defined
    # weights

    # weights from layer 0 to layer 1
    my_mlp.weight_matrices[0].fill(1)

    # weights from layer 1 to layer 2
    my_mlp.weight_matrices[1].fill(1)

    # weights from layer 2 to layer 3
    my_mlp.weight_matrices[2].fill(1)

    # define an input vector
    # and do a feedforward step
    input_vec = np.ones(5)
    print("input_vec=", input_vec)

    # do a feedforward step
    my_mlp.feedforward(input_vec)

    # show neuron output values in each layer
    for i in range(my_mlp.nr_layers):
        my_mlp.show_output(i)

else:

    # test MLP class with large network
    my_mlp.add_layer(100,  TF.identity)
    my_mlp.add_layer(500,  TF.identity)
    my_mlp.add_layer(5000, TF.identity)
    my_mlp.add_layer(500,  TF.identity)
    my_mlp.add_layer(10,   TF.identity)

    my_mlp.weight_matrices[0].fill(1)
    my_mlp.weight_matrices[1].fill(1)
    my_mlp.weight_matrices[2].fill(1)
    my_mlp.weight_matrices[3].fill(1)

    # define an input vector
    # and do a feedforward step
    input_vec = np.ones(100)

    # do a feedforward step
    start = timer()
    for i in range(1):
        my_mlp.feedforward(input_vec)
    end = timer()
    print("Time needed = ",end - start, "sec")

    # show neuron output values in each layer
    #for i in range(my_mlp.nr_layers):
    #    my_mlp.show_output(i)



