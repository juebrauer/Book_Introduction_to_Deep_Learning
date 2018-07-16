import numpy as np
import math


def func_identity(x):
    return x

def func_sigmoid(x):
    return 1 / (1 + math.exp(-x))

def func_relu(x):
    return x if x>0 else 0

def func_squared(x):
    return x*x

def func_ramp(x):
    return 1+0.01*x if x>0 else 0.01*x


def derivative_identity(x):
    return 1



"""
derivative of standard logistic function
f(x) is f'(x) = f(x)*(1-f(x))
see https://en.wikipedia.org/wiki/Logistic_function#Derivative
"""
def derivative_sigmoid(x):
    return func_sigmoid(x) * (1 - func_sigmoid(x))

def derivative_relu(x):
    return 1 if x>0 else 0

def derivative_squared(x):
    return 2*x

def derivative_ramp(x):
    return 0.01


func_identity  = np.vectorize(func_identity)
func_sigmoid   = np.vectorize(func_sigmoid)
func_relu      = np.vectorize(func_relu)
func_squared   = np.vectorize(func_squared)
func_ramp      = np.vectorize(func_ramp)
derivative_identity  = np.vectorize(derivative_identity)
derivative_sigmoid   = np.vectorize(derivative_sigmoid)
derivative_relu      = np.vectorize(derivative_relu)
derivative_squared   = np.vectorize(derivative_squared)
derivative_ramp      = np.vectorize(derivative_ramp)



class TF:
    identity = 1
    sigmoid  = 2
    relu     = 3
    squared  = 4
    ramp     = 5




class mlp:

    nr_layers            = 0
    nr_neurons_per_layer = []
    tf_per_layer         = []
    weight_matrices      = []
    neuron_act_vecs      = []
    neuron_out_vecs      = []

    learn_rate           = 0.01
    neuron_err_vecs      = []
    train_steps_done     = 0

    def __init__(self):
        print("Generated a new empty MLP")


    """
    Returns the output vector of the MLP
    as a NumPy array
    """
    def get_output_vector(self):

        return self.neuron_out_vecs[len(self.neuron_out_vecs)-1]



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
            W = np.random.uniform(low=-1.0, high=1.0,
                                  size=(nr_neurons_before+1,nr_neurons))

            # 2.3 store the new weight matrix
            self.weight_matrices.append(W)

            # 2.4 output some information about the
            #     weight matrix just generated
            print("Generated a new weight matrix W. Shape is",
                  W.shape)
            size = W.nbytes/1024.0
            print("Size of weight matrix in KB"
                  " is {0:.2f}".format(size))


        # 3. generate a new neuron activity and
        # #  neuron output vector
        act_vec = np.zeros(nr_neurons)
        out_vec = np.zeros(nr_neurons)
        err_vec = np.zeros(nr_neurons)
        self.neuron_act_vecs.append( act_vec )
        self.neuron_out_vecs.append( out_vec )
        self.neuron_err_vecs.append( err_vec )

        # 4. update number of layers
        self.nr_layers += 1

        # 5. show current MLP architecture
        self.show_architecture()

        # 6. prepare vector in which we want to
        #    store the average weight change
        #    per layer in order to test for the
        #    vanishing gradient problem
        self.avg_weight_change_per_layer = np.zeros(self.nr_layers)


    """
    Given an input vector, we compute
    the output of all the neurons layer by layer
    into the direction of the output layer
    """
    def feedforward(self, input_vec):

        # 1. set output of neurons from first layer
        #    to input vector values
        N = len(input_vec)
        self.neuron_out_vecs[0] = input_vec

        # 2. now compute neuron outputs layer by layer
        for layer_nr in range(1,self.nr_layers):

            # 2.1 get output vector previously computed
            o = self.neuron_out_vecs[layer_nr-1]

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
            act_mat_this_layer = np.matmul(o_mat,W)

            # 2.6 apply transfer function
            if self.tf_per_layer[layer_nr]==TF.sigmoid:
                out_mat_this_layer =\
                    func_sigmoid(act_mat_this_layer)
            elif self.tf_per_layer[layer_nr]==TF.identity:
                out_mat_this_layer =\
                    func_identity(act_mat_this_layer)
            elif self.tf_per_layer[layer_nr]==TF.relu:
                out_mat_this_layer = \
                    func_relu(act_mat_this_layer)
            elif self.tf_per_layer[layer_nr]==TF.squared:
                out_mat_this_layer = \
                    func_squared(act_mat_this_layer)
            elif self.tf_per_layer[layer_nr]==TF.ramp:
                out_mat_this_layer = \
                    func_ramp(act_mat_this_layer)

            # 2.7 store activity and output of neurons
            self.neuron_act_vecs[layer_nr] = \
                act_mat_this_layer.flatten()
            self.neuron_out_vecs[layer_nr] = \
                out_mat_this_layer.flatten()



    """
    Show output values of all neurons
    in the specified layer
    """
    def show_output(self, layer):

        print("output values of neuron in layer",layer,":",
              self.neuron_out_vecs[layer])


    """
    Shows some statistics about the weights,
    e.g. what is the maximum and the minimum weight in
    each weight matrix
    """
    def show_weight_statistics(self):

        for layer_nr in range(0,self.nr_layers-1):
            W = self.weight_matrices[layer_nr]
            print("Weight matrix for weights from"
                  "layer #",layer_nr,"to layer #",
                  layer_nr+1, ":")
            print("\t shape:", W.shape)
            print("\t min value: ", np.amin(W))
            print("\t max value: ", np.amax(W))
            print("\t W", W)
        print("\n")


    """
    Show state of neurons (activity and output values)
    """
    def show_neuron_states(self):

        for layer_nr in range(0, self.nr_layers):
            print("Layer #", layer_nr)
            print("\t act:", self.neuron_act_vecs[layer_nr])
            print("\t out:", self.neuron_out_vecs[layer_nr])
        print("\n")


    """
    Set a new learn rate which is used in the
    weight update step
    """
    def set_learn_rate(self, new_learn_rate):
        self.learn_rate = new_learn_rate


    """
    Given a pair (input_vec, teacher_vec) we adapt
    the weights of the MLP such that the desired output vector
    (which is the teacher vector)
    is more likely to be generated the next time if the
    input vector is presented as input
    
    Note: this is the Backpropagation learning algorithm!
    """
    def train(self, input_vec, teacher_vec):

        # 1. first do a feedfoward step with the input vector
        self.feedforward(input_vec)

        # 2. first compute the error signals for the output
        #    neurons
        tf_type    = self.tf_per_layer[self.nr_layers-1]
        nr_neurons = self.nr_neurons_per_layer[self.nr_layers-1]
        act_vec    = self.neuron_act_vecs[self.nr_layers-1]
        out_vec    = self.neuron_out_vecs[self.nr_layers-1]
        err_vec    = -(out_vec-teacher_vec)
        if tf_type==TF.sigmoid:
            err_vec *= derivative_sigmoid(act_vec)
        elif tf_type==TF.identity:
            err_vec *= derivative_identity(act_vec)
        elif tf_type==TF.relu:
            err_vec *= derivative_relu(act_vec)
        elif tf_type==TF.squared:
            err_vec *= derivative_squared(act_vec)
        elif tf_type == TF.squared:
            err_vec *= derivative_ramp(act_vec)
        self.neuron_err_vecs[self.nr_layers-1] = err_vec

        # 3. now go from layer N-1 to layer 2 and
        #    compute for each hidden layer the
        #    error signals for each neuron

        # going layer for layer backwards ...
        for layer_nr in range(self.nr_layers-2, 0, -1):

            nr_neurons_this_layer = \
                self.nr_neurons_per_layer[layer_nr]
            nr_neurons_next_layer = \
                self.nr_neurons_per_layer[layer_nr+1]
            W       = self.weight_matrices[layer_nr]
            act_vec = self.neuron_act_vecs[layer_nr]
            tf_type = self.tf_per_layer[layer_nr]

            # run over all neurons in this layer ...
            for neuron_nr in range(0,nr_neurons_this_layer):

                # compute the sum of weighted error signals from
                # neurons in the next layer
                sum_of_weighted_error_signals = 0.0

                # run over all neurons in next layer ...
                for neuron_nr2 in range (0,nr_neurons_next_layer):

                    # get error signal for neuron_nr2 in next layer
                    err_vec = self.neuron_err_vecs[layer_nr+1]
                    err_signal = err_vec[neuron_nr2]

                    # get weight from
                    # neuron_nr  in layer_nr to
                    # neuron_nr2 in layer_nr+1
                    #
                    # Important:
                    # at W[0][neuron_nr2] is the bias
                    # weight to neuron_nr2
                    # at W[1][neuron_nr2] is the first
                    # "real" weight to neuron_nr2
                    weight = W[neuron_nr+1][neuron_nr2]

                    # update sum
                    sum_of_weighted_error_signals +=\
                        err_signal * weight

                # compute and store error signal for
                # neuron with id neuron_nr in this layer
                err_signal = sum_of_weighted_error_signals
                if tf_type == TF.sigmoid:
                    err_signal *= \
                        derivative_sigmoid(act_vec[neuron_nr])
                elif tf_type == TF.identity:
                    err_signal *= \
                        derivative_identity(act_vec[neuron_nr])
                elif tf_type == TF.relu:
                    err_signal *= \
                        derivative_relu(act_vec[neuron_nr])
                elif tf_type == TF.squared:
                    err_signal *= \
                        derivative_squared(act_vec[neuron_nr])
                elif tf_type == TF.ramp:
                    err_signal *= \
                        derivative_ramp(act_vec[neuron_nr])
                self.neuron_err_vecs[layer_nr][neuron_nr] =\
                    err_signal


        # 4. now that we have the error signals for all
        #    neurons (hidden and output neurons) in the net
        #    computed, let's change the weights according to
        #    the weight update formulas
        for layer_nr in range(self.nr_layers - 1, 0, -1):

            nr_neurons_this_layer = \
                self.nr_neurons_per_layer[layer_nr]
            nr_neurons_prev_layer = \
                self.nr_neurons_per_layer[layer_nr-1]

            avg_weight_change_this_layer_this_train_step = 0.0

            nr_weights = 0
            for neuron_nr in range(0, nr_neurons_this_layer):

                # get error signal for this neuron
                err_signal = \
                    self.neuron_err_vecs[layer_nr][neuron_nr]

                for weight_nr in range(0, nr_neurons_prev_layer+1):

                    # get output value of sending neuron
                    out_val_sending_neuron = 1
                    if weight_nr>0:
                        out_val_sending_neuron = \
                            self.neuron_out_vecs[layer_nr-1][weight_nr-1]

                    # compute weight change
                    weight_change = \
                        self.learn_rate * \
                        err_signal * \
                        out_val_sending_neuron

                    # for computing the average weight change in this layer
                    avg_weight_change_this_layer_this_train_step += abs(weight_change)
                    nr_weights += 1

                    self.weight_matrices[layer_nr-1][weight_nr][neuron_nr] += \
                        weight_change

            avg_weight_change_this_layer_this_train_step /= nr_weights

            # update accumulated moving average of weight change
            self.avg_weight_change_per_layer[layer_nr] = \
                (self.avg_weight_change_per_layer[layer_nr] * self.train_steps_done + \
                 avg_weight_change_this_layer_this_train_step) / (self.train_steps_done+1)


        # one more training step done
        self.train_steps_done += 1



    # end of train()



    """
    For checking whether the vanishing gradient problem
    happens it is interesting to see the average weight
    change per layer
    """
    def show_avg_weight_change_per_layer(self):

        print("\nAverage weight change for each layer:")

        for layer_nr in range(self.nr_layers):

            print("Layer #", layer_nr ,
                  ": {0:.15f}".format(self.avg_weight_change_per_layer[layer_nr])
                 )
        print("\n")

    # end show_avg_weight_change_per_layer()


