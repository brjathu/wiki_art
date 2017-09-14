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


def search(content_vect):
    table = []
    class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    for l in class_list:
        search_dir = os.listdir("style/WIKI_STYLE/" + str(l) + "/features/content1000/")
        for file in search_dir:
            cont = scipy.io.loadmat("style/WIKI_STYLE/" + str(l) + "/features/content1000/" + file)['prob']

            sum_val = np.sum(np.sum((content_vect - cont)**2))

            table.append([file, float(sum_val), l])

    a = np.array(table)

    b = np.array(sorted(a, key=lambda a_entry: float(a_entry[1])))

    name = (b[0:2, 0])
    class_label = (b[0:2, 2])

    list_out = []
    for i in range(2):
        # print(class_label[i] + "    " + str(name[i][:-4]) + ".jpg")
        list_out.append(dic[class_label[i] + "    " + str(name[i][:-4]) + ".jpg"])

    return list_out



# preprocessing
name = []
count = 0

class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
class_lable = np.repeat(class_list, 100)

########################################################
#
#         PCA for testing samples
#
########################################################

correct = 0
total = 0
num_test_per_class = 5

result = np.zeros((2, 0))
result = np.array(result)
for l in class_list:
    print("class ==> " + str(l))
    location_style = os.listdir("style/WIKI_STYLE/" + str(l) + "/features/content1000/")
    location_style = random.sample(location_style, num_test_per_class)
    for target_file in location_style:

        qury_string = str(l) + "    " + str(target_file[:-4]) + ".jpg"
        # print(qury_string)

        tag_true = dic[qury_string]
        # print(tag_true)

        content = scipy.io.loadmat("style/WIKI_STYLE/" + str(l) + "/features/content1000/" + target_file)['prob']  # /0.89662

        out = search(content)

        output = np.zeros((2, 1))

        # for i in range(2):
        #     for tag in tag_true:
        #         if(tag in out[i]):
        #             output[i] = output[i] + 1
        # for i in range(2):
        #     if(tag_true[0] in out[i]):
        #         output[i] = output[i] + 1
        for tag in tag_true:
            if(tag in out[1]):
                output[1] = 1
        result = np.hstack([result, output])
        # print(result)
    print(np.sum(result[1:]) / (result.shape[1]) * 100)


print(np.sum(result[1:]) / (result.shape[1]) * 100)


# if(total != 0):
# print(correct / total * 100)

#         if (int(out) == l):
#             correct = correct + 1
#         total = total + 1

# print(correct / total * 100)

# map = confusion_matrix(y_true, y_pred)

# print(map)

# df_cm = pd.DataFrame(map, index = [i for i in class_list],
#                   columns = [i for i in class_list])
# plt.figure(figsize = (10,7))
# sn.heatmap(df_cm, annot=True)
# plt.show()
