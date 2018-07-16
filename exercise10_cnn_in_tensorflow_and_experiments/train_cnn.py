# file: train_cnn.py
#
# A CNN implementation in TensorFlow
#
# First set the variable <dataset_root>
# It tells the Python script where your training and
# validation data is.
#
# Store your training images in a sub-folder for each
# class in <dataset_root>/train/
# e.g.
#    <dataset_root>/train/bikes
#    <dataset_root>/train/cars
#
# Store your test images in a sub-folder for each
# class in <dataset_root>/test/
# e.g.
#    <dataset_root>/test/bikes
#    <dataset_root>/test/cars
#
# Then let the script run.
# The final trained model will be saved.

from dataset_reader import dataset
import tensorflow as tf

# Experiments
# Exp-Nr   Comment
# 01       10 hidden layers:
#          INPUT->C1->P1->C2->P2->C3->C4->C5->P3->FC1->FC2->OUT
#          trained 100 mini-batches of size 32
#          feature maps: 10-15-20-25-30
#
# 02       10 hidden layers:
#          INPUT->C1->P1->C2->P2->C3->C4->C5->P3->FC1->FC2->OUT
#          trained 1000 mini-batches of size 32
#          feature maps: 10-15-20-25-30
#
# 03       4 hidden layers: INPUT->C1->P1->FC1->FC2->OUT
#          trained 1000 mini-batches of size 32
#          feature maps: 10

exp_nr = 1
dataset_root =\
    "V:/01_job/12_datasets/imagenet/cars_vs_bikes_prepared/"


# helper function to build 1st conv layer with filter size 11x11
# and stride 4 (in both directions) and no padding
def conv1st(name, l_input, filter, b):
    cov = tf.nn.conv2d(l_input, filter,
                       strides=[1, 4, 4, 1], padding='VALID')
    return tf.nn.relu(tf.nn.bias_add(cov, b), name=name)


# in all other layers we use a stride of 1 (in both directions)
# and a padding such that the spatial dimension (width,height)
# of the output volume is the same as the spatial dimension
# of the input volume
def conv2d(name, l_input, w, b):
    cov = tf.nn.conv2d(l_input, w,
                       strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(tf.nn.bias_add(cov, b), name=name)

# generates a max pooling layer
def max_pool(name, l_input, k, s):
    return tf.nn.max_pool(l_input,
                        ksize=[1, k, k, 1],
                        strides=[1, s, s, 1],
                        padding='VALID', name=name)


# helper function to generate a CNN
def build_cnn_model(_X, keep_prob, n_classes, imagesize, img_channel):
    # prepare matrices for weights
    _weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, img_channel, 10])),
        'wc2': tf.Variable(tf.random_normal([5, 5, 10, 15])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 15, 20])),
        'wc4': tf.Variable(tf.random_normal([3, 3, 20, 25])),
        'wc5': tf.Variable(tf.random_normal([3, 3, 25, 30])),
        'wd1': tf.Variable(tf.random_normal([6 * 6 * 30, 40])),
        'wd2': tf.Variable(tf.random_normal([40, 40])),
        'out': tf.Variable(tf.random_normal([40, n_classes])),
        'exp3_wd1': tf.Variable(tf.random_normal([27 * 27 * 10, 40]))
    }

    # prepare vectors for biases
    _biases = {
        'bc1': tf.Variable(tf.random_normal([10])),
        'bc2': tf.Variable(tf.random_normal([15])),
        'bc3': tf.Variable(tf.random_normal([20])),
        'bc4': tf.Variable(tf.random_normal([25])),
        'bc5': tf.Variable(tf.random_normal([30])),
        'bd1': tf.Variable(tf.random_normal([40])),
        'bd2': tf.Variable(tf.random_normal([40])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # reshape input picture
    _X = tf.reshape(_X, shape=[-1, imagesize, imagesize, img_channel])

    if (exp_nr == 1 or exp_nr==2):

        # feature hierarchy:
        # topology:
        # INPUT->C1->P1->C2->P2->C3->C4->C5->P3->FC1->FC2->OUT
        conv1 =\
            conv1st('conv1', _X, _weights['wc1'], _biases['bc1'])
        pool1 =\
            max_pool('pool1', conv1, k=3, s=2)
        conv2 =\
            conv2d('conv2', pool1, _weights['wc2'], _biases['bc2'])
        pool2 =\
            max_pool('pool2', conv2, k=3, s=2)
        conv3 =\
            conv2d('conv3', pool2, _weights['wc3'], _biases['bc3'])
        conv4 =\
            conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'])
        conv5 =\
            conv2d('conv5', conv4, _weights['wc5'], _biases['bc5'])
        pool3 =\
            max_pool('pool3', conv5, k=3, s=2)

        # classifier:
        # fully connected layers
        dense1 =\
            tf.reshape(pool3,
                       [-1, _weights['wd1'].get_shape().as_list()[0]])
        dense1 =\
            tf.nn.relu(tf.matmul(dense1,
                       _weights['wd1']) + _biases['bd1'],
                       name='fc1')
        dense2 =\
            tf.nn.relu(tf.matmul(dense1,
                                 _weights['wd2']) + _biases['bd2'],
                                 name='fc2')
        out =\
            tf.matmul(dense2, _weights['out']) + _biases['out']

    elif exp_nr == 3:

        # feature hierarchy:
        # topology:
        # INPUT->C1->P1->FC1->FC2->OUT
        conv1 =\
            conv1st('conv1', _X, _weights['wc1'], _biases['bc1'])
        print("conv1 shape: ", conv1.get_shape())
        pool1 =\
            max_pool('pool1', conv1, k=3, s=2)
        print("pool1 shape: ", pool1.get_shape())

        # classifier:
        # fully connected layers
        dense1 =\
            tf.reshape(pool1, [-1, 27*27*10])
        dense1 =\
            tf.nn.relu(
                   tf.matmul(dense1,
                             _weights['exp3_wd1']) + _biases['bd1'],
                             name='fc1')
        print("dense1 shape: ", dense1.get_shape())
        dense2 =\
            tf.nn.relu(
                   tf.matmul(dense1,
                             _weights['wd2']) + _biases['bd2'],
                             name='fc2')
        print("dense2 shape: ", dense2.get_shape())

        out =\
            tf.matmul(dense2, _weights['out']) + _biases['out']
        print("out shape: ", out.get_shape())

    return [out, _weights['wc1']]


# 1. create a training and testing Dataset object that stores
#    the training / testing images
training = dataset(dataset_root + "train", ".jpeg")
testing = dataset(dataset_root + "validation", ".jpeg")

# 2. set training parameters
learn_rate = 0.001
batch_size = 32
display_step = 1
if exp_nr==1:
    nr_mini_batches_to_train = 100
elif exp_nr==2:
    nr_mini_batches_to_train = 1000
elif exp_nr==3:
    nr_mini_batches_to_train = 1000

save_filename = 'save/model.ckpt'
logs_path = './logfiles'

n_classes = training.num_labels
dropout = 0.8  # dropout (probability to keep units)
imagesize = 227
img_channel = 3

x = tf.placeholder(tf.float32, [None,
                                imagesize,
                                imagesize,
                                img_channel])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

[pred, filter1st] = build_cnn_model(x,
                                    keep_prob,
                                    n_classes,
                                    imagesize,
                                    img_channel)
cost = tf.reduce_mean(
         tf.nn.softmax_cross_entropy_with_logits(logits=pred,
                                                 labels=y))
# cost = tf.reduce_mean(tf.squared_difference(pred, y))

global_step = tf.Variable(0, trainable=False)

optimizer = tf.train.AdamOptimizer(
              learning_rate=learn_rate).minimize(
                 cost, global_step=global_step)
# optimizer = tf.train.GradientDescentOptimizer(lr).
#                minimize(cost, global_step=global_step)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()
tf.add_to_collection("x", x)
tf.add_to_collection("y", y)
tf.add_to_collection("keep_prob", keep_prob)
tf.add_to_collection("pred", pred)
tf.add_to_collection("accuracy", accuracy)

print("\n\n")
print("----------------------------------------")
print("I am ready to start the training...")
print("So I will train a CNN, starting with a"+
      "learn rate of", learn_rate)
print("I will train ", nr_mini_batches_to_train,
      "mini batches of ", batch_size, "images")
print("Your input images will be resized to ",
      imagesize, "x", imagesize, "pixels")
print("----------------------------------------")

with tf.Session() as my_session:
    my_session.run(tf.global_variables_initializer())

    step = 1
    while step < nr_mini_batches_to_train:

        batch_ys, batch_xs = training.nextBatch(batch_size)
        # note: batch_ys and batch_xs are tuples each
        # batch_ys a tuple of e.g. 32 one-hot NumPy arrays
        # batch_xs a tuple of e.g. 32 NumPy arrays of shape
        #  (width, height, 3)


        _ = my_session.run([optimizer],
                            feed_dict={x: batch_xs,
                                       y: batch_ys,
                                       keep_prob: dropout})

        if step % display_step == 0:
            acc = my_session.run(accuracy,
                                 feed_dict={x: batch_xs,
                                            y: batch_ys,
                                            keep_prob: 1.})
            loss = my_session.run(cost, feed_dict={x: batch_xs,
                                                   y: batch_ys,
                                                   keep_prob: 1.})
            print("learn rate:" + str(learn_rate) +
                  " mini batch:" + str(step) +
                  ", minibatch loss= " + "{:.5f}".format(loss) +
                  ", batch accuracy= " + "{:.5f}".format(acc))
        step += 1

    print("\n")
    print("Training of CNN model finished.")

    save_filename =\
        "saved_model_exp0" + str(exp_nr) + "/final_model.ckpt"
    saver.save(my_session, save_filename, global_step=step)
    print("Saved CNN model to file '",save_filename,"'")
