'''
Minimalistic exmaple of a MLP in TensorFlow
---
by Prof. Dr. Juergen Brauer, www.juergenbrauer.org
'''

import numpy as np
from data_generator import data_generator
import tensorflow as tf
import cv2
from timeit import default_timer as timer


# Do you want to start the MLP with the same weights
# each time? E.g., for comparing different optimizers?
# Then activate the following code line:
tf.set_random_seed(12345)



# test data parameters
WINSIZE = 600
NR_CLUSTERS = 5
NR_SAMPLES_TO_GENERATE = 10000

# MLP parameters
NR_EPOCHS = 2000

# for RELU transfer function use smaller learn rate
# than for logistic transfer function
# Also use more hidden neurons! (e.g. 2-30-12-2)
#LEARN_RATE = 0.001

# for logistic transfer function
LEARN_RATE = 0.5

MINI_BATCH_SIZE = 100
NR_NEURONS_INPUT   = 2
NR_NEURONS_HIDDEN1 = 50 # nr of neurons in 1st hidden layer
NR_NEURONS_HIDDEN2 = 25  # nr of neurons in 2nd hidden layer
NR_NEURONS_OUTPUT  = 2

# store 2D weight matrices & 1D bias vectors for all
# neuron layers in two dictionaries
weights = {
    'h1': tf.Variable(tf.random_normal(
        [NR_NEURONS_INPUT, NR_NEURONS_HIDDEN1])),
    'h2': tf.Variable(tf.random_normal(
        [NR_NEURONS_HIDDEN1, NR_NEURONS_HIDDEN2])),
    'out': tf.Variable(tf.random_normal(
        [NR_NEURONS_HIDDEN2, NR_NEURONS_OUTPUT]))
}
biases = {
    'b1': tf.Variable(tf.random_normal(
        [NR_NEURONS_HIDDEN1])),
    'b2': tf.Variable(tf.random_normal(
        [NR_NEURONS_HIDDEN2])),
    'out': tf.Variable(tf.random_normal(
        [NR_NEURONS_OUTPUT]))
}


# visualization parameters
RADIUS_SAMPLE = 3
COLOR_CLASS0 = (255,0,0)
COLOR_CLASS1 = (0,0,255)
NR_TEST_SAMPLES = 10000

# for saving images
image_counter = 0


'''
helper function to create a 4 layer MLP
input-layer --> 
  hidden layer #1 --> 
    hidden layer #2 --> 
      output layer
'''
def multilayer_perceptron(x, weights, biases):

    # hidden layer #1 with RELU
    layer_1 = tf.add(tf.matmul(x, weights['h1']),
                     biases['b1'])
    #layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.sigmoid(layer_1)

    # hidden layer #2 with RELU
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']),
                     biases['b2'])
    #layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.sigmoid(layer_2)

    # output layer with linear activation (no RELUs!)
    out_layer = tf.matmul(layer_2, weights['out'])\
                + biases['out']

    # return the MLP model
    return out_layer



def generate_and_show_training_data():

    #  1. generate training data
    my_dg = data_generator()
    data_samples = \
        my_dg.generate_samples_two_class_problem(
            NR_CLUSTERS,
            NR_SAMPLES_TO_GENERATE)
    nr_samples = len(data_samples)

    # 2. generate empty image for visualization
    #    initialize image with white pixels (255,255,255)
    img = np.ones((WINSIZE, WINSIZE, 3), np.uint8) * 255

    # 3. visualize positions of samples
    for i in range(nr_samples):

        # 3.1 get the next data sample
        next_sample = data_samples[i]

        # 3.2 get input and output vector
        #     (which are both NumPy arrays)
        input_vec = next_sample[0]
        output_vec = next_sample[1]

        # 3.3 prepare a tupel from the NumPy input vector
        sample_coord = (int(input_vec[0] * WINSIZE),
                        int(input_vec[1] * WINSIZE))

        # 3.4 get class label from output vector
        if output_vec[0] > output_vec[1]:
            class_label = 0
        else:
            class_label = 1
        color = (0, 0, 0)
        if class_label == 0:
            color = COLOR_CLASS0
        elif class_label == 1:
            color = COLOR_CLASS1

        # 3.4
        cv2.circle(img, sample_coord, RADIUS_SAMPLE, color)

    # 4. show visualization of samples
    cv2.imshow('Training data', img)
    c = cv2.waitKey(1)
    cv2.imwrite("V:/tmp/training_data.png", img)

    return data_samples


def visualize_decision_boundaries(
        the_session, epoch_nr, x_in, mlp_output_vec):

    global image_counter

    NR_TEST_SAMPLES = 10000

    # 1. prepare a large input matrix
    input_mat = np.zeros((NR_TEST_SAMPLES,NR_NEURONS_INPUT))
    np.random.seed(1)
    for i in range(NR_TEST_SAMPLES):

        # generate random coordinate in [0,1) x [0,1)
        rnd_x = np.random.rand()
        rnd_y = np.random.rand()

        # prepare an input vector
        input_vec = np.array([rnd_x, rnd_y])

        # set corresponding line in input matrix
        input_mat[i,:] = input_vec


    # 2. do a feedforward step for all test vectors
    #    in the input matrix
    res = the_session.run(mlp_output_vec,
                          feed_dict={x_in: input_mat})


    # 3. draw each sample in predicted class color

    # generate empty white color image
    img = np.ones((WINSIZE, WINSIZE, 3), np.uint8) * 255
    #print(res)
    for i in range(NR_TEST_SAMPLES):

        # get the input vector back from matrix
        input_vec = input_mat[i,:]

        # get the output vector from result tensor
        output_vec = res[i,:]

        # now get the predicted class from the output
        # vector
        class_label = 0 if output_vec[0] > output_vec[1] else 1

        # map class label to a color
        color = COLOR_CLASS0 if class_label == 0 else COLOR_CLASS1

        # draw circle
        sample_coord = (int(input_vec[0] * WINSIZE),
                        int(input_vec[1] * WINSIZE))
        cv2.circle(img, sample_coord, RADIUS_SAMPLE, color)



    # 4. finaly show the image
    cv2.rectangle(img, (WINSIZE - 120, 0), (WINSIZE - 1, 20),
                  (255, 255, 255), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,
                "epoch #" + str(epoch_nr).zfill(3),
                (WINSIZE - 110, 15), font, 0.5, (0, 0, 0), 1,
                cv2.LINE_AA)
    cv2.imshow('Decision boundaries of trained MLP', img)
    c = cv2.waitKey(1)

    # 5. save that image?
    if True:
        filename = "V:/tmp/img_{0:0>4}".format(image_counter)
        image_counter +=1
        cv2.imwrite(filename + ".png", img)


def build_TF_graph():

    # 1. prepare placeholders for the input and output values

    # the input is a 2D matrix:
    # in each row we store one input vector
    x_in = tf.placeholder("float")

    # the output is a 2D matrix:
    # in each row we store one output vector
    y_out = tf.placeholder("float")

    # 2. now the use helper function defined before to
    #    generate a MLP
    mlp_output_vec = multilayer_perceptron(x_in, weights, biases)

    # 3. define a loss function
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(my_mlp, y))
    loss = \
        tf.reduce_mean(tf.squared_difference(mlp_output_vec, y_out))

    # 4. add an optimizer to the graph
    # optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)


    use_opt_nr = 5

    if use_opt_nr==1:
        optimizer =\
            tf.train.GradientDescentOptimizer(LEARN_RATE)
    elif use_opt_nr==2:
        optimizer = \
            tf.train.MomentumOptimizer(learning_rate=LEARN_RATE,
                                       momentum=0.9,
                                       use_nesterov = True)
    elif use_opt_nr==3:
        optimizer = \
            tf.train.AdagradOptimizer(learning_rate=0.01)
    elif use_opt_nr==4:
        optimizer = \
            tf.train.RMSPropOptimizer(learning_rate=0.0001,
                                      decay=0.9)
    elif use_opt_nr==5:
        optimizer = \
            tf.train.AdamOptimizer(learning_rate=0.001,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=10e-8)
    optimizer = optimizer.minimize(loss)

    # 5. create a summary to track current
    #    value of loss function
    tf.summary.scalar("Value-of-loss", loss)

    # 6. in case we want to track multiple summaries
    #    merge all summaries into a single operation
    summary_op = tf.summary.merge_all()

    return optimizer, mlp_output_vec, loss, x_in, y_out



def MLP_training(data_samples, optimizer, mlp_output_vec,
                 loss, x_in, y_out):

    NR_SAMPLES = len(data_samples)

    with tf.Session() as my_session:

        # initialize all variables
        my_session.run(tf.global_variables_initializer())
        fw = tf.summary.FileWriter("V:/tmp/summary", my_session.graph)

        # how many mini batches do we have to process?
        nr_batches_to_process = \
            int(NR_SAMPLES / MINI_BATCH_SIZE)

        # in each epoch all training samples will be presented
        for epoch_nr in range(0, NR_EPOCHS):

            print("Training MLP. Epoch nr #", epoch_nr)

            # in each mini batch some of the training samples
            # will be feed-forwarded, the weight changes for a
            # single sample will be computed and all weight changes
            # be accumulated for all samples in the mini-batch.
            # Then the weights will be updated.
            start = timer()
            for mini_batch_nr in range(0, nr_batches_to_process):

                # a) generate list of indices
                sample_indices = np.arange(0, NR_SAMPLES)

                # b) shuffle the indices list
                sample_indices = np.random.shuffle(sample_indices)

                # c) now prepare a matrix
                #    with one sample input vector in each row and
                #    another matrix with the corresponding desired
                #    output vector in each row
                input_matrix =\
                    np.zeros((MINI_BATCH_SIZE, NR_NEURONS_INPUT))
                output_matrix =\
                    np.zeros((MINI_BATCH_SIZE, NR_NEURONS_OUTPUT))
                startpos = mini_batch_nr * MINI_BATCH_SIZE
                row_counter = 0
                for next_sample_id in \
                        range(startpos, startpos + MINI_BATCH_SIZE):
                    # get next training sample from dataset class
                    # the dataset is a list of lists
                    # in each list entry there are two vectors:
                    # the input vector and the output vector
                    next_sample = data_samples[next_sample_id]

                    # get input and output vector
                    # (which are both NumPy arrays)
                    input_vec = next_sample[0]
                    output_vec = next_sample[1]

                    # copy input vector to respective
                    # row in input matrix
                    input_matrix[row_counter, :] = input_vec

                    # copy output vector respective
                    # row in output matrix
                    output_matrix[row_counter, :] = output_vec

                    row_counter += 1

                # d) run the optimizer node --> training will happen
                #    now the actual feed-forward step and the
                #    computations will happen!
                _, curr_loss = my_session.run(
                    [optimizer, loss],
                    feed_dict={x_in: input_matrix,
                               y_out: output_matrix})

                # print("current loss for mini-batch=", curr_loss)

            # after each epoch:
            # visualize the decision boundaries of the MLP
            # trained so far
            end = timer()
            print("Time needed to train one epoch: ",
                  end - start, "sec")

            print("Now testing the MLP...")
            visualize_decision_boundaries(my_session,
                                          epoch_nr,
                                          x_in,
                                          mlp_output_vec)


def main():

    data_samples = generate_and_show_training_data()

    optimizer, mlp_output_vec, loss, x_in, y_out = build_TF_graph()

    MLP_training(data_samples, optimizer, mlp_output_vec, loss,
                 x_in, y_out)

    print("End of MLP TensorFlow test.")

main()