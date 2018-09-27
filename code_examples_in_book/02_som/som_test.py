'''
File: som_test.py

Data samples are generated, then a Self-Organizing Map (SOM).
The SOM is updated in each step with a randomly chosen data
sample.
Positions of samples, the "positions" (= weights) of the SOM
neurons and the 2D topology is visualized after each step.
'''

import sys
import cv2
import numpy as np
from data_generator import data_generator
from som import som

# SOM parameters
INPUT_DIM = 2
# square root of NR_NEURONS should be an integer!
NR_NEURONS = 7*7
LEARN_RATE = 0.2
adapt_neighbors = False
WAIT_FOR_KEYPRESS = True

# visualization parameters
HEIGHT = 600
WIDTH  = 600
COLOR_SAMPLE         = (128,128,128)
COLOR_CURRENT_SAMPLE = (255,0,0)
COLOR_NEURON         = (0,0,255)
COLOR_NEIGHBORHOOD   = (0,128,255)
RADIUS_SAMPLES = 5
RADIUS_NEURONS = 5

# set this to a higher value to
# speed up training dramatically!
VISUALIZE_EACH_N_STEPS = 20



print("Your Python version is: " + sys.version)
print("Your OpenCV version is: " + cv2.__version__)


print("************")
print("* SOM Test *")
print("************")
print("Press 's' to generate new samples.")
print("Press 'i' to initialize neurons to grid.")
print("Press 'j' to initialize neurons to (0,0).")
print("Press 'a' to swith on/off neighbor adaptation.")



# 1. generate samples
my_dg = data_generator(WIDTH, HEIGHT)
NR_CLUSTERS = 15
NR_SAMPLES_PER_CLUSTER = 30
data_samples = my_dg.generate_samples_near_to_clusters(
    NR_CLUSTERS, NR_SAMPLES_PER_CLUSTER)
nr_samples = len(data_samples)
print("Type of data_samples is ", type(data_samples))
print("There are ", nr_samples, "samples in the list.")


# 2. generate a SOM
my_som = som(INPUT_DIM, NR_NEURONS)
my_som.initialize_neuron_weights_to_grid([10, 10, 150,150])



# 3. SOM training
while (True):

    # 3.1 retrieve randomly a sample vector
    rnd_vec_id = np.random.randint(nr_samples)
    vec = data_samples[rnd_vec_id]


    # 3.2 train the SOM with this vector
    my_som.train( vec, LEARN_RATE, adapt_neighbors )


    # 3.3 start visualization section?
    if (my_som.nr_steps_trained % VISUALIZE_EACH_N_STEPS != 0):
        continue

    # 3.4 generate empty image for visualization
    img = np.ones((HEIGHT, WIDTH, 3), np.uint8) * 255


    # 3.5 visualize positions of samples
    for i in range(nr_samples):

        # get the next sample vector (it is an NumPy array)
        # convert it to a tuple (which is need as coordinates
        # for the circle draw command)
        sample_coord = tuple(data_samples[i])
        cv2.circle(img, sample_coord,
                   RADIUS_SAMPLES, COLOR_SAMPLE)

    # 3.6 visualize position of current input sample
    cv2.circle(img, tuple(vec),
               RADIUS_SAMPLES, COLOR_CURRENT_SAMPLE, 3)


    # 3.7 visualize positions of all neurons
    for i in range(NR_NEURONS):

        # get the neurons weight vector and
        # convert it to a tuple
        neuron_coord =\
            tuple( (my_som.list_neurons[i].weight_vec).
                   astype(int) )
        cv2.circle(img, neuron_coord,
                   RADIUS_NEURONS, COLOR_NEURON, 2)


    # 3.8 visualize neighborhood relationship of neurons
    #    by drawing a line between each two neighbored
    #    neurons
    for i in range(NR_NEURONS):

        # prepare the neuron's coordinates as a tuple
        # (for drawing coords)
        neuron_i_coord =\
            tuple((my_som.list_neurons[i].weight_vec).
                  astype(int))

        # now get a list of all neighbors of this neuron
        neighbors = my_som.get_neighbors(i)
        # print("Neighbors of neuron ",i," are: ", neighbors)

        # for all neighbors of this neuron:
        for j in neighbors:

            # prepare the neuron's coordinates as a tuple
            neuron_j_coord = \
                tuple((my_som.list_neurons[j].weight_vec).
                      astype(int))

            # draw a line between neuron i and
            # its neighbored neuron j
            cv2.line(img, neuron_i_coord, neuron_j_coord,
                     COLOR_NEIGHBORHOOD, 1)

    # 3.9 show how many steps we have already trained
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,
                str(my_som.nr_steps_trained).zfill(5),
                (WIDTH-60, 20), font, 0.5, (0,0,0), 1,
                cv2.LINE_AA)


    # 3.10 show visualization of samples, neuron locations
    #      and neuron neighborhood relations
    cv2.imshow('img', img)


    # 3.11 wait for a key
    if WAIT_FOR_KEYPRESS:
        c = cv2.waitKey(0)
    else:
        c = cv2.waitKey(1)


    # 3.12 generate new sample distribution?
    #      interesting to see, how an existing SOM adapts
    #      to new input data!
    if c!=-1:

        # start with new samples?
        if chr(c)=='s':
            print("Generating new samples")
            data_samples = \
                 my_dg.generate_samples_near_to_clusters\
                (NR_CLUSTERS, NR_SAMPLES_PER_CLUSTER)

        # reinitialize neuron weights to grid?
        if chr(c)=='i':
            print("Reinitialization of neuron weightsto grid")
            my_som.initialize_neuron_weights_to_grid(
                    [100, 100, WIDTH - 200, HEIGHT - 200])

        # reinitialize neuron weights to (0,0)?
        if chr(c) == 'j':
            print("Reinitialization of neuron weights to (0,0)")
            my_som.initialize_neuron_weights_to_origin()

        # switch on/off neighbor adaptation
        if chr(c)=='a':
            adapt_neighbors = not adapt_neighbors
            print("Adapt neighbors? --> ", adapt_neighbors )


    # 3.13 save current image?
    #      e.g. for later animation of SOM adaption process
    #      in a video or animated gif
    if False:
        if (my_som.nr_steps_trained == 1 or
            my_som.nr_steps_trained % 100 == 0):
            cv2.imwrite("V:/tmp/" +
                str(my_som.nr_steps_trained).zfill(4) + ".png", img)

