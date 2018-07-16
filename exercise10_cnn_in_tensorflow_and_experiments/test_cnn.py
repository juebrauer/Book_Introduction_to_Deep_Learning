# file: test.py
#
# Restores a learned CNN model and tests it.
#
# Store your test images in a sub-folder for each
# class in <dataset_root>/validation/
# e.g.
#    <dataset_root>/validation/bikes
#    <dataset_root>/validation/cars

# 
# ---
# Prof. Dr. Juergen Brauer, www.juergenbrauer.org

from dataset_reader import dataset

dataset_root =\
    "V:/01_job/12_datasets/imagenet/cars_vs_bikes_prepared/"
testing  = dataset(dataset_root + "validation", ".jpeg")

import tensorflow as tf
import numpy as np

# Parameters
batch_size = 1

ckpt = tf.train.get_checkpoint_state("saved_model_exp01")
saver = tf.train.import_meta_graph(
          ckpt.model_checkpoint_path + '.meta')

pred = tf.get_collection("pred")[0]
x = tf.get_collection("x")[0]
keep_prob = tf.get_collection("keep_prob")[0]

sess = tf.Session()
saver.restore(sess, ckpt.model_checkpoint_path)

# test
step_test = 0
correct=0
while step_test * batch_size < len(testing):

    # testing_ys and testing_xs are tuples
    testing_ys, testing_xs = testing.nextBatch(batch_size)

    # get first image and first ground truth vector
    # from the tuples
    first_img             = testing_xs[0]
    first_groundtruth_vec = testing_ys[0]

    # at first iteration:
    # show shape of image and ground truth vector
    if step_test == 0:
        print("Shape of testing_xs is :",
              first_img.shape)
        print("Shape of testing_ys is :",
              first_groundtruth_vec.shape)

    # given the input image,
    # let the CNN predict the category!
    predict = sess.run(pred,
                       feed_dict={x: testing_xs,
                                  keep_prob: 1.})

    # get ground truth label and
    # predicted label from output vector
    groundtruth_label = np.argmax(first_groundtruth_vec)
    predicted_label = np.argmax(predict, 1)[0]

    print("\nImage test: ", step_test)
    print("Ground truth label    :",
          testing.label2category[groundtruth_label])
    print("Label predicted by CNN:",
          testing.label2category[predicted_label])

    if predicted_label == groundtruth_label:
        correct+=1
    step_test += 1

print("\n---")
print("Classified", correct, "images correct of",
                    step_test,"in total.")
print("Classification rate in" +
      "percent = {0:.2f}".format(correct/step_test*100.0))

