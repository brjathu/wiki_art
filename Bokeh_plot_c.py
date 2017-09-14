from bokeh.plotting import figure, output_file, show
from bokeh.models.widgets import Panel, Tabs, Slider, Toggle
from bokeh.layouts import widgetbox, row, column, layout
from bokeh.models import HoverTool, Range1d, CustomJS, ColumnDataSource
import bokeh.plotting as bp
import gensim
from math import log10
import pickle
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models.doc2vec import TaggedDocument
from gensim import models
import scipy.misc
import scipy.io
import random
import numpy as np
import os

# arrange files
# location_content = os.listdir("/media/brjathu/Jathu1/DATA61/content1000/")
# list_of_random_items = random.sample(location_content, 5000)
# for file in list_of_random_items:
#     print(file)
#     os.system("cp /media/brjathu/Jathu1/DATA61/content1000/"+ file + " ../icons/features/content1000/"+ file )
#     os.system("cp /media/brjathu/Jathu1/DATA61/icons/sample_icons/"+ file[0:-4]+".png" + " ../icons/raw/"+ file[0:-4] +".png" )


loadFromPickle = False
# location_content = os.listdir("../icons/features/content1000/")                                         # location of app icons directory


pickleFile = "save_c.pickle"  # location of pickle file to save t-SNE vectors after calculation


if loadFromPickle:
    (vectors, names) = pickle.load(open(pickleFile, "rb"))
    print("Loaded from Pickle")
else:
    vectors = []
    names = []
    icons = []

    class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                  14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    class_label = np.repeat(class_list, 100)
    for l in class_list:
        location = os.listdir("style/WIKI_STYLE/" + str(l) + "/features/content4096/")
        for file in location:
            cnt = (scipy.io.loadmat("style/WIKI_STYLE/" + str(l) + "/features/content4096/" + file)['f6'])
            vectors.append(cnt[0])
            icons.append("style/WIKI_STYLE/" + str(l) + "/img/" + file[0:-4] + ".jpg")
            names.append(file)
    vec = np.array(vectors, dtype=np.float32)

    tSNEModel = TSNE(n_components=2, random_state=0, metric="cosine", perplexity=50)
    pca = PCA(n_components=2, random_state=0)
    print(vec.shape)

    vectors = tSNEModel.fit_transform(vec)

    print(vec.shape)

    print("tSNE done")
    pickle.dump((vectors, names), open(pickleFile, "wb"))
    print("saved pickle")

x = [a[0] for a in vectors]
y = [a[1] for a in vectors]


def convert_to_hex(colour):
    red = int(colour[0] * 255)
    green = int(colour[1] * 255)
    blue = int(colour[2] * 255)
    return '#{r:02x}{g:02x}{b:02x}'.format(r=red, g=green, b=blue)


colourClas = [convert_to_hex(cm.gist_rainbow((int(class_list.index(i)) / 41.01))) for i in class_label]
# print(colourClas)

source1 = bp.ColumnDataSource({"Name": names,
                               "Class": class_label,
                               'x_tSNE': vectors[:, 0],
                               'y_tSNE': vectors[:, 1],
                               'color': colourClas,
                               'imgs': icons})

output_file("Apps_tSNE.html")

title = "Icons"
p1 = figure(plot_width=1500, plot_height=900, title=title, tools="pan,box_zoom,reset,hover")

p1.scatter(x='x_tSNE', y='y_tSNE', source=source1, color='color')

hover1 = p1.select(dict(type=HoverTool))


htmlLayout = """
        <div style="width:40vw">
            <div>
            <img
                src="@imgs" height="64" alt="@imgs" width="64"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
                <span style="font-size: 17px; font-weight: bold;">@Name</span>
                <span style="font-size: 15px; color: #966;">@Class</span>
            </div>
        </div>
        """
hover1.tooltips = htmlLayout


tab1 = Panel(child=p1, title="Density")
tabs = Tabs(tabs=[tab1])

show(tabs)
