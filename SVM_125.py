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
from PIL import Image
import pickle

# preprocessing

train = False
name = []
count = 0

class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
class_lable = np.repeat(class_list, 100)

mean_c = np.load("style/content_mean.npy")
mean_s = np.load("style/style_mean.npy")
ratio = 5
max = 0


if (train == True):
    for l in class_list:
        print(l)
        location = os.listdir("style/WIKI_STYLE/" + str(l) + "/features/combined125")

        for file in location:

            comb = scipy.io.loadmat("style/WIKI_STYLE/" + str(l) + "/features/combined125/" + file)['combined']

            if (count == 0):
                final_vector = np.reshape(comb, (1, -1))
            else:
                final_vector = np.vstack([final_vector, np.reshape(comb, (1, -1))])
            count = count + 1

    # np.save("style/style_mean.npy" , np.mean(style_all, axis = 0))
    # np.save("style/content_mean.npy" , np.mean(content_all, axis = 0))

    # Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create SVM classification object
    clf = SVC(kernel='linear', gamma="auto", decision_function_shape="ovo")
    # there is various option associated with it, like changing kernel, gamma and C value. Will discuss more
    # about it in next section.Train the model using the training sets and check score
    clf.fit(final_vector, class_lable.T)

    with open('svm.pkl', 'wb') as f:
        pickle.dump(clf, f)


else:
    ########################################################
    #
    #         PCA for testing samples
    #
    ########################################################

    with open('svm.pkl', 'rb') as f:
        clf = pickle.load(f)

    correct = 0
    total = 0
    num_test_per_class = 50
    y_true = np.repeat(class_list, num_test_per_class)
    y_pred = []

    for l in class_list:
        print(l)
        location_style = os.listdir("style/WIKI_STYLE_TEST/" + str(l) + "/features/combined125/")
        location_style = random.sample(location_style, num_test_per_class)
        for file in location_style:

            comb = scipy.io.loadmat("style/WIKI_STYLE_TEST/" + str(l) + "/features/combined125/" + file)['combined']

            test = np.reshape(comb, (1, -1))

            # print(test.shape)
            res = clf.predict(test)
            print(res[0], l)
            y_pred.append(res[0])

            if(total != 0):
                print(correct / total * 100)

            if (res[0] == l):
                correct = correct + 1
            total = total + 1

    print(correct / total * 100)

    map = confusion_matrix(y_true, y_pred)

    print(map)

    df_cm = pd.DataFrame(map, index=[i for i in class_list],
                         columns=[i for i in class_list])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
