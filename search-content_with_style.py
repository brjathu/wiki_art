from sklearn import svm
import numpy as np
import os
import scipy.misc
import scipy.io
from sklearn.svm import SVC
import random
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
import pickle


with open('tag-5-label.pkl', 'rb') as f:
    dic = pickle.load(f)


style_sqr_matrix = np.load("norm/style_sqr_matrix.npy")
content_sqr_matrix = np.load("norm/content_sqr_matrix_4096.npy")
name_list = np.load("norm/name.npy")
class_list_main = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
class_lable_main = np.reshape(np.repeat(class_list_main, 100), (2500, 1))


def search(name, gamma):
    ind = np.where(name_list == name)
    res = np.hstack([np.reshape(name_list, (2500, 1)), (content_sqr_matrix[ind].T + gamma * style_sqr_matrix[ind].T), class_lable_main])
    b = np.array(sorted(res, key=lambda a_entry: float(a_entry[-2])))
    return(b[0:4])


class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

correct = 0
total = 0
num_test_per_class = 5

graph = []
gamma_array = range(20)
for gamma in gamma_array:
    print(gamma)
    average_tag = 0
    average_class = 0
    for test in range(3):
        result = np.zeros((4, 2))
        result = np.array(result)
        for l in class_list:
            # print("class ==> " + str(l))
            location = os.listdir("style/WIKI_STYLE/" + str(l) + "/features/content1000/")
            location = random.sample(location, num_test_per_class)
            for target_file in location:

                qury_string = str(l) + "    " + str(target_file[:-4]) + ".jpg"
                # print(qury_string)

                tag_true = dic[qury_string]

                out = search(target_file, 1)
                output_tag = np.zeros((4, 2))
                # print(out)
                count = 0
                for q in out:
                    # print(q)
                    tag_result = dic[q[-1] + "    " + str(q[0][:-4]) + ".jpg"]
                    for tag in tag_true:
                        if(tag in tag_result):
                            output_tag[count][0] = 1
                            break

                    if(str(l) == q[-1]):
                        output_tag[count][1] = 1

                    count = count + 1

                result = np.vstack([result, output_tag])

        dim = result[4:, 0].shape
        accuracy_tag = np.sum(result[4:, 0]) / (dim) * 100
        # print("accuracy tag  => ", accuracy_tag[0])

        dim = result[4:, 1].shape
        accuracy_class = np.sum(result[4:, 1]) / (dim) * 100
        # print("accuracy class => ", accuracy_class[0])

        average_tag = average_tag + accuracy_tag
        average_class = average_class + accuracy_class
    graph.append([average_tag / 3, average_class / 3])
    print(average_class / 3)

np.save("norm/graph.npy", graph)
