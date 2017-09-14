import os
import numpy as np
import tensorflow as tf
import vgg19
import utils
import scipy.io
import time
import math


tf.logging.set_verbosity(tf.logging.ERROR)

location = os.listdir("/flush1/raj034/wiki/")
for l in location:
    print(l)
    class_name = l
    batch_size = 100

    location_class = os.listdir("/flush1/raj034/wiki/" + str(class_name) + "/")
    os.system("mkdir /flush1/raj034/wiki/features/" + str(class_name) + "/content1000")
    os.system("mkdir /flush1/raj034/wiki/features/" + str(class_name) + "/content4096")
    os.system("mkdir /flush1/raj034/wiki/features/" + str(class_name) + "/style")
    # print(location)

    count = 0
    name = []
    for image in location_class:
        name.append(image)
        # print(name)
        if(count % batch_size == 0):
            img1 = utils.load_image("/flush1/raj034/wiki/" + str(class_name) + "/" + image)
            batch = img1.reshape((1, 224, 224, 3))
        else:
            img1 = utils.load_image("/flush1/raj034/wiki/" + str(class_name) + "/" + image)
            batch1 = img1.reshape((1, 224, 224, 3))
            batch = np.concatenate((batch, batch1), 0)

        print(batch.shape)

        if ((count + 1) % batch_size == 0):
            g = tf.Graph()
            with g.as_default(), g.device('/gpu'), tf.Session() as sess:
                images = tf.placeholder("float", [batch_size, 224, 224, 3])
                feed_dict = {images: batch}
                vgg = vgg19.Vgg19()
                with tf.name_scope("content_vgg"):
                    vgg.build(images)
                prob, conv5_1, content4096 = sess.run([vgg.prob, vgg.conv5_1, vgg.fc6], {feed_dict=feed_dict})

                for i in range(batch_size):
                    features = np.reshape(conv5_1[i], (-1, 512))
                    gram = np.matmul(features.T, features) / features.size
                    scipy.io.savemat("/flush1/raj034/wiki/features/" + str(class_name) + "/content4096/" + name[int(((count + 1) / batch_size - 1) * batch_size + i)][0:-4] + '.mat', mdict={'f6': fc6[i]}, oned_as='row')
                    scipy.io.savemat("/flush1/raj034/wiki/features/" + str(class_name) + "/style/" + name[int(((count + 1) / batch_size - 1) * batch_size + i)][0:-4] + '.mat', mdict={'conv5_1': gram}, oned_as='row')
                    scipy.io.savemat("/flush1/raj034/wiki/features/" + str(class_name) + "/content1000/" + name[int(((count + 1) / batch_size - 1) * batch_size + i)][0:-4] + '.mat', mdict={'prob': prob[i]}, oned_as='row')

        count = count + 1
