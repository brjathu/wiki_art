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
EPOCH = 2000
LR = 0.001         # learning rate

class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

train_loss = []

mean_c = np.load("style/content_mean.npy")
mean_s = np.load("style/style_mean.npy")

tf.reset_default_graph()
# tf placeholder
tf_x = tf.placeholder(tf.float32, [None, 2000])    # value in the range of (0, 1)

# encoder
en0 = tf.layers.dense(tf_x, 1000, tf.nn.relu)
en1 = tf.layers.dense(en0, 500, tf.nn.relu)
en2 = tf.layers.dense(en1, 250, tf.nn.relu)
encoded = tf.layers.dense(en2, 125, tf.nn.relu)

# decoder
de0 = tf.layers.dense(encoded, 250, tf.nn.relu)
de1 = tf.layers.dense(de0, 500, tf.nn.relu)
de2 = tf.layers.dense(de1, 1000, tf.nn.relu)
decoded = tf.layers.dense(de2, 2000, tf.nn.tanh)

loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
train = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

for epoch in range(EPOCH):
    for l in class_list:
        count = 0

        location = os.listdir("style/WIKI_STYLE/" + str(l) + "/features/style1000/")

        for file in location:
            style_file = scipy.io.loadmat("style/WIKI_STYLE/" + str(l) + "/features/style1000/" + file)['conv5_1']
            content_file = scipy.io.loadmat("style/WIKI_STYLE/" + str(l) + "/features/content1000/" + file)['prob'] - mean_c

            if(count == 0):
                features = np.hstack([content_file, (np.reshape(style_file, (1, -1)) - mean_s) / 7826.146])
            else:

                features1 = np.hstack([content_file, (np.reshape(style_file, (1, -1)) - mean_s) / 7826.146])
                features = np.vstack([features, features1])

            count = count + 1
        b_x = features
        _, loss_ = sess.run([train, loss], {tf_x: b_x})

    if epoch % 1 == 0:
        print("epoch ===> " + str(epoch) + '       train loss: %.10f' % loss_)
        train_loss.append(loss_)
        # plotting decoded image (second row)
        # decoded_data = sess.run(decoded, {tf_x: view_data})

saver.save(sess, 'train/2000/my_test_model', write_meta_graph=False)

plt.xlabel("EPOCH")
plt.ylabel("Training loss")
plt.title("Traing Loss Graph")

plt.plot(range(EPOCH), train_loss)
plt.show()


np.save("loss.npy", train_loss)
