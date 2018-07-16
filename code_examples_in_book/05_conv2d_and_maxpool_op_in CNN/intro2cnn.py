'''
Code to show effect of TensorFlow's
- conv2d() operator
- max_pool() operator

***
by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org
'''

import numpy as np
import cv2
import tensorflow as tf


def load_two_test_images():

    # oad two test images and
    # convert int values to float values in [0,1]
    img0 = cv2.imread("test0.jpg")
    img1 = cv2.imread("test1.jpg")
    #print("max value of img0 is", img0.max())
    img0 = img0.astype(np.float32)/255.0
    img1 = img1.astype(np.float32)/255.0
    cv2.imshow("img0", img0)
    #cv2.imshow("img1", img1)
    print("img0 has shape", img0.shape)
    print("img1 has shape", img1.shape)
    print("img0 has dtype", img0.dtype)

    return img0,img1



def conv2d_demo(minibatch):

    height   = minibatch.shape[1]
    width    = minibatch.shape[2]
    channels = minibatch.shape[3]

    # create a filter array with two filters:
    # filter / kernel tensor has to have shape
    # [filter_height, filter_width, in_channels, out_channels]
    filters = np.ones(shape=(5,5,channels,2), dtype=np.float32) * -1.0/60.0
    filters[:,2,:,0] = 1.0/15.0 # vertical line (in all 3 channels)
    filters[2,:,:,1] = 1.0/15.0 # horizontal line (in all 3 channels)

    # create TF graph
    X = tf.placeholder(tf.float32, shape=(None,height,width,channels))
    convop = tf.nn.conv2d(X,
                          filters,
                          strides=[1,2,2,1],
                          padding="SAME")

    # filter both images (mini batch)
    # by running the convolution op in the graph
    with tf.Session() as my_session:
        convresult = my_session.run(convop, feed_dict={X:minibatch})

    print("shape of convresult is", convresult.shape)

    convres_img0_filt0 = convresult[0,:,:,0]
    convres_img0_filt1 = convresult[0,:,:,1]

    print("max/min value of convres_img0_filt0 is",
          convres_img0_filt0.max(), convres_img0_filt0.min())

    convres_img0_filt0 = convres_img0_filt0 /  convres_img0_filt0.max()
    convres_img0_filt1 = convres_img0_filt1 /  convres_img0_filt1.max()

    print("max/min value of convres_img0_filt0"
          " after normalization is now",
          convres_img0_filt0.max(), convres_img0_filt0.min())

    cv2.imshow("conv result img0 filter0", convres_img0_filt0)
    cv2.imshow("conv result img0 filter1", convres_img0_filt1)

    cv2.imwrite("conv result img0 filter0.png", convres_img0_filt0*255)
    cv2.imwrite("conv result img0 filter1.png", convres_img0_filt1*255)

    return convresult


def maxpool_demo(minibatch):

    height = minibatch.shape[1]
    width = minibatch.shape[2]
    channels = minibatch.shape[3]

    # create TF graph
    X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    maxpool_op = tf.nn.max_pool(X,
                                ksize=[1,2,2,1],
                                strides=[1,2,2,1],
                                padding="VALID")

    # filter both images (mini batch)
    # by running the convolution op in the graph
    with tf.Session() as my_session:
        maxpool_result = my_session.run(maxpool_op, feed_dict={X: minibatch})

    print("shape of maxpool_result is", maxpool_result.shape)

    poolres_img0_chan0 = maxpool_result[0, :, :, 0]

    print("max/min value of poolres_img0_chan0 is",
          poolres_img0_chan0.max(), poolres_img0_chan0.min())

    poolres_img0_chan0 = poolres_img0_chan0 / poolres_img0_chan0.max()

    print("max/min value of poolres_img0_chan0"
          " after normalization is now",
          poolres_img0_chan0.max(), poolres_img0_chan0.min())

    cv2.imshow("poolres_img0_chan0", poolres_img0_chan0)

    #cv2.imshow("max_pooled image as color image", maxpool_result[0, :,:,:])


def main():

    # 1. load two test images
    img0, img1 = load_two_test_images()

    # 2. put the two test images into a mini-batch
    #    so the mini-batch is a 4D array of dimension:
    #   (nr-of-images, img-height, img-width, nr-img-channels)
    height = img0.shape[0]
    width = img0.shape[1]
    channels = img0.shape[2]
    minibatch = np.zeros([2,height,width,channels], dtype=np.float32)
    print("minibatch has shape",minibatch.shape)
    minibatch[0,:,:,:] = img0
    minibatch[1,:,:,:] = img1

    # 3. filter each image in the mini-batch with
    #    two different 3D filters
    convresult = conv2d_demo(minibatch)


    # 4. now create a graph where we process the mini-batch
    #    with the max pool operation
    maxpool_demo(convresult)

    # 5. the show is over. wait for a key press.
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
