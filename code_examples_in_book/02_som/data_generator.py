"""
File: data_generator.py

In this file a class data_generator is defined that
allows to generate data sample distributions of different
forms.

It is meant to be a helper class for the SOM test.
We want to test and see how the SOM adapts to different
sample distributions.
"""

import numpy as np


class data_generator:

    img_width  = 0
    img_height = 0

    def __init__(self, width, height):
        print("A new data generator object has been created.")
        self.img_width  = width
        self.img_height = height


    def generate_samples_near_to_clusters(self, nr_clusters, nr_samples_per_cluster):

        SPACE = 100
        CLUSTER_RADIUS = 75


        # 1. create <nr_clusters> random 2D cluster centers
        #    with <nr_samples_per_cluster> random samples per cluster
        data_samples = []
        for i in range(nr_clusters):

            # generate random cluster center
            center_x = SPACE + np.random.randint(self.img_width-2*SPACE)
            center_y = SPACE + np.random.randint(self.img_height-2*SPACE)

            for j in range(nr_samples_per_cluster):

                # compute random offset vector to cluster center
                rnd_offset_x = np.random.randint(CLUSTER_RADIUS)
                rnd_offset_y = np.random.randint(CLUSTER_RADIUS)

                # compute final 2D sample coordinates
                x = center_x + rnd_offset_x
                y = center_y + rnd_offset_y

                # is the sample within the image dimension?
                if (x<0): x=0
                if (y<0): y=0
                if (x>self.img_width) : x = self.img_width
                if (y>self.img_height): y = self.img_height

                # store the sample coordinate in the list
                data_samples.append( np.array([x,y]) )

        return data_samples

