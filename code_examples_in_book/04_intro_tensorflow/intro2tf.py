import tensorflow as tf


def separator():
    print("\n==================\n")

def show_tf_version():
    print("Your TF version is", tf.__version__)
    print(type(tf.__version__))

def show_available_tf_devices():
    from tensorflow.python.client import device_lib
    devices = [x.name for x in device_lib.list_local_devices()]
    print(devices)

def types_of_tensors():
    a = tf.Variable(3, name='var1')
    b = tf.Variable([1.0,2.0,3.0], name='var2')
    c = tf.Variable([[1,0],[0,1]], name='var3')
    print(a)
    print(b)
    print(c)

    d = tf.constant(3.14159, name='pi')
    print(d)

    e = tf.placeholder(tf.float32, shape=[28, 28], name="myPlaceholder-input-image")
    print(e)


def a_first_simple_graph():

    a = tf.Variable(3.0)
    b = tf.Variable(4.0)
    c = tf.multiply(a, b)
    print(c)
    with tf.Session() as my_session:
        my_session.run(tf.global_variables_initializer())
        resulting_tensor = my_session.run(c)
        print("resulting tensor=",resulting_tensor)


def a_simple_graph_with_placeholders():

    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b  # shortcut for tf.add(a, b)
    with tf.Session() as my_session:
        my_session.run(tf.global_variables_initializer())
        print(my_session.run(adder_node, {"c":3, b:4} ))
        print(my_session.run(adder_node, {a: [1,2], b: [3,4]} ))
        print(my_session.run(adder_node, {a: [[1, 1],[1,1]], b: [[1, 1],[0,1]]} ))


def tf_is_lazy():

    a = tf.Variable(2)
    b = tf.Variable(5)
    c = tf.multiply(a,b)
    d = tf.add(a,c)
    print(d)
    with tf.Session() as my_session:
        my_session.run(tf.global_variables_initializer())
        d_value = my_session.run(d)
        print ("d_value=",d_value)


def saving_variables():

    a = tf.Variable(3)
    b = tf.Variable(15, name="variable-b")
    saver = tf.train.Saver()
    print("type of save is ",type(saver))
    with tf.Session() as my_session:
        my_session.run(tf.global_variables_initializer())
        print("a:", my_session.run(a))
        print("b:", my_session.run(b))


def loading_variables():

    a = tf.Variable(0)
    b = tf.Variable(0, name="variable-b")
    saver = tf.train.Saver()
    print("type of save is ",type(saver))
    with tf.Session() as my_session:

        # restore a session
        saver.restore(my_session, "V:/tmp/my_model.ckpt")
        print("a:", my_session.run(a))
        print("b:", my_session.run(b))


def visualizing_the_graph():

    a = tf.Variable(3, name="var-a")
    b = tf.Variable(4, name="var-b")
    c = tf.Variable(5, name="var-c")
    d = tf.multiply(a,b, name="op-multiply")
    e = tf.add(c,d, name="op-add")
    with tf.Session() as my_session:
        my_session.run(tf.global_variables_initializer())
        print (my_session.run(d))
        fw = tf.summary.FileWriter("V:/tmp/summary", my_session.graph)


def visualizing_the_graph_using_scopes():

    with tf.name_scope("All_my_variables") as scope:
        a = tf.Variable(3, name="var-a")
        b = tf.Variable(4, name="var-b")
        c = tf.Variable(5, name="var-c")

    with tf.name_scope("My_operations") as scope:

        with tf.name_scope("Some_additions") as scope:
            aplusb = tf.add(a, b, name="a-plus-b")
            bplusc = tf.add(b, c, name="b-plus-c")

        with tf.name_scope("Some_multiplications") as scope:
            asquared = tf.multiply(a, a, name="a-squared")
            csquared = tf.multiply(c, c, name="c-squared")

        with tf.name_scope("Final_computation") as scope:
            compresult = tf.add(asquared,csquared, "final-add")

    with tf.Session() as my_session:
        my_session.run(tf.global_variables_initializer())
        print(my_session.run(compresult))
        fw = tf.summary.FileWriter("V:/tmp/summary", my_session.graph)


def visualizing_variable_values_using_tb():

    import random

    # the value of a will be incremented by some
    # placeholder value
    a = tf.Variable(42, name="var-a")
    rndnumber_placeholder = \
        tf.placeholder(tf.int32, shape=[], name="rndnumber_placeholder")
    update_node = tf.assign(a,tf.add(a, rndnumber_placeholder))

    # create a summary to track value of a
    tf.summary.scalar("Value-of-a", a)

    # in case we want to track multiple summaries
    # merge all summaries into a single operation
    summary_op = tf.summary.merge_all()

    with tf.Session() as my_session:
        my_session.run(tf.global_variables_initializer())
        fw = tf.summary.FileWriter("V:/tmp/summary", my_session.graph)

        # generate random numbers that are used
        # as values for the placeholder
        for step in range(500):

            rndnum = int(-10 + random.random() * 20)
            new_value_of_a = \
                my_session.run(update_node,
                    feed_dict={rndnumber_placeholder: rndnum})

            print("new_value_of_a=", new_value_of_a)

            # compute summary
            summary = my_session.run(summary_op)

            # add merged summaries to filewriter,
            # this will save the data to the file
            fw.add_summary(summary, step)


def training_a_first_simple_model():
    # A first simple TensorFlow learner
    #
    # Given some random samples of 2D data points (x,y),
    # the following TF script tries to find a line
    # that best describes the data points in terms of
    # a minimal sum of squared errors (SSE)

    import tensorflow as tf
    import numpy as np

    # 1.1 create 1D array with 100 random numbers drawn uniformly from [0,1)
    x_in          = tf.placeholder("float")
    y_teacher_out = tf.placeholder("float")

    NR_SAMPLES = 100
    NR_EPOCHS = 10
    train_data_in = np.random.rand(NR_SAMPLES)
    print("\nx_data:", train_data_in)
    print("data type:", type(train_data_in))

    # 1.2 now compute the y-points
    train_data_out = train_data_in * 1.2345 + 0.6789

    # so now we have ground truth samples (x,y)
    # and the TF learner will have to estimate the line parameters
    # y=W*x+b with W=1.2345 and b=0.6789
    #
    # These parameters are called variables in TF.


    # 2. We initialize them with a random W and b=0
    W            = tf.Variable(3.0)
    b            = tf.Variable(0.0)
    y_actual_out = tf.add(tf.multiply(W,x_in), b)

    # 3.1 Now define what to optimize at all:
    #     here we want to minimize the SSE.
    #     This is our "loss" of a certain line model
    #     Note: this is just another node in the
    #     computation graph that computes something
    loss_func = tf.square(y_teacher_out - y_actual_out)

    # 3.2 We can use different optimizers in TF for model learning
    my_optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss_func)


    with tf.Session() as my_session:

        my_session.run(tf.global_variables_initializer())

        # 4. Print inial value of W and b
        print("\n")
        print("initial W", my_session.run(W))
        print("initial b", my_session.run(b))

        # 5. Train the model:
        print("\n")
        for epoch_nr in range(NR_EPOCHS):

            # do a gradient step for each training sample
            # individually!
            # i.e. this is Stochastic Gradient Descent with
            # one training sample only
            for sample_nr in range(NR_SAMPLES):

                # Do another gradient descent step to come to a better
                # W and b
                my_session.run(my_optimizer,
                               feed_dict={x_in         : train_data_in[sample_nr],
                                          y_teacher_out: train_data_out[sample_nr]})

                # Show current value of W and b
                print(epoch_nr, my_session.run(W), my_session.run(b))


def main():

    #separator()
    #show_tf_version()

    #separator()
    #show_available_tf_devices()

    #separator()
    #types_of_tensors()

    #separator()
    #a_first_simple_graph()

    #separator()
    #a_simple_graph_with_placeholders()

    #separator()
    #tf_is_lazy()

    #separator()
    #saving_variables()

    #separator()
    #loading_variables()

    #separator()
    #visualizing_the_graph()

    #separator()
    #visualizing_the_graph_using_scopes()

    #separator()
    #visualizing_variable_values_using_tb()

    #separator()
    training_a_first_simple_model()

main()