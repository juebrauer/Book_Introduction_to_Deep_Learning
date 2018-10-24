# File: perceptron_classifier_exercise05.py
#
# Perceptron that learns to classify handwritten digits
#
# The MNIST dataset http://yann.lecun.com/exdb/mnist/
# consists of 28x28 pixel images:
#   - 55.000 training images
#   - 10.000 test images

# for random numbers
from random import randint

# we will use OpenCV to display the data
import cv2

# we will use TensorFlow to access the MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# for working with N-dimensional arrays
import numpy as np

# for plotting the classification rate on test data during training
import matplotlib.pyplot as plt




'''
The ramp transfer function
'''
def f(x):
    if (x<=0):
        return 0
    else:
        return 1

f = np.vectorize(f)


'''
Donwload & unpack the MNIST data
Also prepare direct access to data matrices:
x_train, y_train, x_test, y_test
'''
def read_mnist_data():

    # 1. download and read data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # 2. show data type of the mnist object
    print("type of mnist is ", type(mnist))

    # 3. show number of training and test examples
    print("There are ", mnist.train.num_examples, " training examples available.")
    print("There are ", mnist.test.num_examples, " test examples available.")

    # 4. prepare matrices (numpy.ndarrays) to access the training / test images and labels
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels
    print("type of x_train is", type(x_train))
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_test: ", x_test.shape)
    print("y_test: ", y_test.shape)

    return x_train,y_train,x_test,y_test



'''
This function will show n random example images
from the training set, visualized using OpenCV's imshow() function
'''
def show_some_mnist_images(n, x_train, y_train):

    nr_train_examples = x_train.shape[0]

    for i in range(0,n):

        # 1. guess a random number between 0 and 55.000-1
        rnd_number = randint(0, nr_train_examples)

        # 2. get corresponding output vector
        correc_out_vec = y_train[rnd_number,:]
        print("Here is example MNIST image #",i," It is a: ", np.argmax(correc_out_vec))

        # 3. get first row of 28x28 pixels = 784 values
        row_vec = x_train[rnd_number, :]
        print("type of row_vec is ", type(row_vec))
        print("shape of row_vec is ", row_vec.shape)

        # 4. reshape 784 dimensional vector to 28x28 pixel matrix M
        M = row_vec.reshape(28, 28)

        # 5. resize image
        M = cv2.resize(M, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

        # 6. show that matrix using OpenCV
        cv2.imshow('image', M)

        # wait for a key
        c = cv2.waitKey(0)

        cv2.destroyAllWindows()


'''
Generate a weight matrix of dimension (nr-of-inputs, nr-of-outputs)
and train the weights according to the Perceptron learning rule
using random sample patterns <input,desired output> from the MNIST
training dataset
'''
def generate_and_train_perceptron_classifier(nr_train_steps,x_train,y_train,x_test,y_test):

    data_to_plot_x = []
    data_to_plot_y = []

    nr_train_examples = x_train.shape[0]

    # 1. generate Perceptron with random weights
    weights = np.random.rand(785, 10)

    # 2. do the desired number of training steps
    for train_step in range(0, nr_train_steps):

        # 2.1 show that we are alive from time to time ...
        if (train_step % 100 == 0):
            print("train_step = ", train_step, end=" -> ")

            correct_rate, nr_correct, nr_wrong = test_perceptron(weights, x_test, y_test)
            correct_percentage = round(correct_rate * 100.0, 2)
            print("classification rate: ", correct_percentage, "%")

            data_to_plot_x.append( train_step )
            data_to_plot_y.append( correct_percentage )

            # Set title of the plot
            #fig = plt.figure("Learning curve")
            #fig.suptitle('Learning curve', fontsize=20)

            plt.plot(data_to_plot_x, data_to_plot_y, 'r')
            plt.ylim(0.0, 100.0)
            plt.grid(True)
            plt.xlabel('train_step', fontsize=14)
            plt.ylabel('classification rate', fontsize=14)

            plt.show(block=False)
            plt.pause(0.001)

            # save image of the current plot
            #fig = plt.gcf()
            #fig.savefig('exercise05_classification_rate_plotting.png')


        # 2.2 choose a random image
        rnd_number = randint(0, nr_train_examples-1)
        input_vec = x_train[rnd_number, :]
        # add bias input "1"
        input_vec = np.append(input_vec, [1])
        input_vec = input_vec.reshape(1, 785)


        # 2.3 compute Perceptron output. Should have dimensions 1x10
        act = np.matmul(input_vec, weights)
        out_mat = f(act)


        # 2.4 compute difference vector
        teacher_out_mat = y_train[rnd_number, :]
        teacher_out_mat = teacher_out_mat.reshape(1, 10)
        # print("shape of teacher_out_vec is ", teacher_out_mat.shape)
        diff_mat = teacher_out_mat - out_mat
        # print("diff_mat = ",diff_mat)


        # 2.5 correct weights
        learn_rate = 0.01
        for neuron_nr in range(0, 10):

            # 2.5.1 get neuron error
            neuron_error = diff_mat[0, neuron_nr]

            # 2.5.2 for all weights to the current neuron <neuron_nr>
            for weight_nr in range(0, 785):
                # get input_value x_i
                x_i = input_vec[0, weight_nr]

                # compute weight change
                delta_w_i = learn_rate * neuron_error * x_i

                # add weight change to current weight
                weights[weight_nr, neuron_nr] += delta_w_i


    # 3. learning has finished. Return the result: the weight matrix
    return weights



'''
Now test how good the Perceptron can classify
on data never seen before, i.e., the test data
'''
def test_perceptron(weights, x_test, y_test):

    nr_test_examples = x_test.shape[0]

    # 1. initialize counters
    nr_correct = 0
    nr_wrong = 0

    # 2. forward all test patterns,
    #    then compare predicted label with ground truth label
    #    and check whether the prediction is right or not
    for test_vec_nr in range(0, nr_test_examples):

        # 2.1 get the test vector
        input_vec = x_test[test_vec_nr, :]
        # add bias input "1"
        input_vec = np.append(input_vec, [1])
        input_vec = input_vec.reshape(1, 785)

        # 2.2 get the desired output vector
        teacher_out_mat = y_test[test_vec_nr, :]
        teacher_out_mat = teacher_out_mat.reshape(1, 10)
        teacher_class = np.argmax(teacher_out_mat)

        # 2.3 compute the actual output of the Perceptron
        act = np.matmul(input_vec, weights)
        out_mat = f(act)
        actual_class = np.argmax(out_mat)

        # 2.4 is the desired class and the actual class the same?
        if (teacher_class == actual_class):
            nr_correct += 1
        else:
            nr_wrong += 1

    # 3. return the test results
    correct_rate = float(nr_correct) / float(nr_correct+nr_wrong)
    return correct_rate, nr_correct, nr_wrong


'''
Here is a simple Nearest Neighbour classifier.
Just store the training data as a database.
If a new vector has to be classified,
just look into the database and search for the most similar vector.
Then take as the predicted class, the corresponding label of that vector.
'''
def NN_classifier(vec_to_classify, x_train, y_train):

    nr_train_examples = x_train.shape[0]

    smallest_diff = 0
    best_matching_vec_nr = 0

    for train_vec_nr in range(0, nr_train_examples):

        # get the next vector from the training database
        train_vec = x_train[train_vec_nr, :]
        train_vec = train_vec.reshape(1, 784)

        # compute distance between the two vectors
        # <vec_to_classify> and <train_vec>
        diff = np.sum(np.abs(train_vec - vec_to_classify))

        # found smaller difference
        if (train_vec_nr==0 or diff<smallest_diff):
            smallest_diff = diff
            best_matching_vec_nr = train_vec_nr

    # get label of best matching vector
    train_out_vec = y_train[best_matching_vec_nr, :]
    train_out_vec = train_out_vec.reshape(1, 10)
    predicted_label = np.argmax(train_out_vec)

    return predicted_label



'''
Let's compare the Perceptron with a very simple classifier:
The nearest neighbour classifier.
'''

def test_NN_classifier(x_train, y_train, x_test, y_test):

    nr_test_examples = x_test.shape[0]

    # 1. initialize counters
    nr_correct = 0
    nr_wrong = 0

    # 2. test all test patterns,
    #    then compare predicted label with ground truth label
    #    and check whether the prediction is right or not
    #for test_vec_nr in range(0, nr_test_examples):
    for test_vec_nr in range(0, 500):

        if (test_vec_nr % 100 == 0):
            print("tested", test_vec_nr, "of the", nr_test_examples, "many test vectors so far...")

        # 2.1 get the test vector
        input_vec = x_test[test_vec_nr, :]
        input_vec = input_vec.reshape(1, 784)

        # 2.2 get the desired output vector
        teacher_out_mat = y_test[test_vec_nr, :]
        teacher_out_mat = teacher_out_mat.reshape(1, 10)
        teacher_class = np.argmax(teacher_out_mat)

        # 2.3 get classification result of NN classifier
        predicted_class = NN_classifier(input_vec, x_train, y_train)

        # 2.4 is the desired class and the actual class the same?
        if (teacher_class == predicted_class):
            nr_correct += 1
        else:
            nr_wrong += 1

    # 3. return the test results
    correct_rate = float(nr_correct) / float(nr_correct + nr_wrong)
    return correct_rate, nr_correct, nr_wrong




def main():

    print("\n1. Get the data")
    x_train,y_train,x_test,y_test = read_mnist_data()

    print("\n2. Show the data")
    show_some_mnist_images(5, x_train, y_train)

    print("\n3. Train Perceptron")
    weights = generate_and_train_perceptron_classifier(5000, x_train, y_train, x_test, y_test)

    print("\n4. Final test of Perceptron")
    correct_rate, nr_correct, nr_wrong = test_perceptron(weights, x_test, y_test)
    print(correct_rate*100.0,"% of the test patterns were correctly classified.")
    print("correctly classified: ", nr_correct)
    print("wrongly classified: ", nr_wrong)

    #print("\n5. Comparison with own classifier")
    #correct_rate, nr_correct, nr_wrong = test_NN_classifier(x_train, y_train, x_test, y_test)
    #print(correct_rate * 100.0, "% of the test patterns were correctly classified.")
    #print("correctly classified: ", nr_correct)
    #print("wrongly classified: ", nr_wrong)

    print("Program end.")


main()
