# Copyright (c) 2017, Networks Group, Data61, CSIRO
#
# Confidential
#
# Jathushan.Rajasegaran@data61.csiro.au


import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import scipy.misc
import scipy.io
import matplotlib.pyplot as plt


tf.set_random_seed(1)
tf.logging.set_verbosity(tf.logging.ERROR)

# Hyper Parameters
BATCH_SIZE = 5
EPOCH = 100
LR = 0.001         # learning rate

class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

mean_c = np.load("style/content_mean.npy")
mean_s = np.load("style/style_mean.npy")

comp = np.load("style/comp.npy")
mean_pca = np.load("style/mean.npy")


tf.reset_default_graph()

# tf placeholder
tf_x = tf.placeholder(tf.float32, [None, 2000])    # value in the range of (0, 1)

# encoder
en0_ = tf.layers.dense(tf_x, 1000, tf.nn.relu)
en1_ = tf.layers.dense(en0_, 500, tf.nn.relu)
en2_ = tf.layers.dense(en1_, 250, tf.nn.relu)
encoded_ = tf.layers.dense(en2_, 125, tf.nn.relu)

# decoder
de0_ = tf.layers.dense(encoded_, 250, tf.nn.relu)
de1_ = tf.layers.dense(de0_, 500, tf.nn.relu)
de2_ = tf.layers.dense(de1_, 1000, tf.nn.relu)
decoded_ = tf.layers.dense(de2_, 2000, tf.nn.tanh)

loss_ = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded_)
# train = tf.train.AdamOptimizer(LR).minimize(loss_)

sess = tf.Session()
# First let's load meta graph and restore weights
saver = tf.train.Saver()  # define a saver for saving and restoring
saver.restore(sess, 'train/2000/my_test_model')


for l in class_list:
    count = 0
    print(l)
    location = os.listdir("style/WIKI_STYLE_TEST/" + str(l) + "/features/style/")

    for file in location:
        style_file = scipy.io.loadmat("style/WIKI_STYLE_TEST/" + str(l) + "/features/style/" + file)['conv5_1']
        content_file = scipy.io.loadmat("style/WIKI_STYLE_TEST/" + str(l) + "/features/content1000/" + file)['prob'] - mean_c

        style1000 = ((np.reshape(style_file, (1, -1)) - mean_pca).dot(comp.T) - mean_s) / 7826.146

        if(count == 0):
            features = np.hstack([content_file, style1000])
        else:

            features1 = np.hstack([content_file, style1000])

            features = np.vstack([features, features1])

        count = count + 1
    b_x = features
    encodedX, lossX = sess.run([encoded_, loss_], {tf_x: b_x})
    print(encodedX.shape)
    i = 0
    for file in location:
        scipy.io.savemat("style/WIKI_STYLE_TEST/" + str(l) + "/features/combined125/" + file, mdict={'combined': encodedX[i]}, oned_as='row')
        i = i + 1
