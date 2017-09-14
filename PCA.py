# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import os
import random
import numpy as np
import scipy.misc
import scipy.io
import math
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
import scipy.io
# from matplotlib.mlab import PCA


def pca2(data, pc_count=None):
    return PCA(n_components=4096).fit(data).transform(data)


def main():
    class_list = [0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    name = []
    count = 0

    # comp = np.load("../style/comp.npy")
    # mean = np.load("../style/mean.npy")

    # print(comp.shape)
    # print(mean.shape)

    #########################################################
    #
    #          PCA for training samples
    #
    #########################################################
    # for l in class_list:
    #     print(l)
    #     location_style = os.listdir("../style/WIKI_STYLE/" + str(l) + "/features/style/")
    #     for file in location_style:
    #         name.append(file)
    #         style = scipy.io.loadmat("../style/WIKI_STYLE/" + str(l) + "/features/style/" + file)

    #         style1 = style['conv5_1']

    #         if(count == 0):
    #             features = np.reshape(style1, (1, -1))
    #         else:

    #             features1 = np.reshape(style1, (1, -1))
    #             features = np.vstack([features, features1])
    #         count = count + 1
    # features = np.load("norm/normalized_style.npy")
    # print(features.shape)
    name = np.load("norm/name.npy")
    # results = pca2(features)

    # # #     # print(results)
    # print(results.shape)
    # # print(results[0].shape)
    # np.save("results.npy", results)
    results = np.load("results.npy")
    print(count)
    num = 0
    for l in class_list:
        print(l)
        for i in range(100):
            scipy.io.savemat('style/WIKI_STYLE/' + str(l) + '/features/style4096/' + name[num][0:-4] + '.mat', mdict={'conv5_1': results[num]}, oned_as='row')
            num = num + 1
    #     count = count + 1

    #########################################################
    #
    #          PCA for testing samples
    #
    #########################################################
    # for l in class_list:
    #     print(l)
    #     location_style = os.listdir("../style/WIKI_STYLE_TEST/"+str(l)+"/features/style/")
    #     location_style = random.sample(location_style, 50)
    #     for file in location_style:

    #         style = scipy.io.loadmat("../style/WIKI_STYLE_TEST/"+str(l)+"/features/style/"+file)

    #         style1 = style['conv5_1']
    #         features = np.reshape(style1 , (1, -1))
    #         out = (features-mean).dot(comp.T)

    #         print(out.shape)

    # print(features.shape)

    # results = pca2(features)

    # #     # print(results)
    # print(results.shape)
    # print(results[0].shape)
    # np.save("results.npy", results)
    # print(count)
    # num = 0
    # for l in class_list:
    #     for i in range(100):
    #         scipy.io.savemat('../style/WIKI_STYLE_TEST/'+str(l)+'/features/style1000/'+name[num][0:-4]+'.mat', mdict={'conv5_1': results[num] }, oned_as='row')
    #         num = num + 1
    # #     count = count + 1


if __name__ == '__main__':
    main()
