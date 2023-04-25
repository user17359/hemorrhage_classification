import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

labels = pd.read_csv("hemorrhage_labels.csv")

data = dict()
data['label'] = []
data['data'] = []   

for i in labels["Patient_no"]:
    # data from CSV
    data['label'].append(labels["Hemorrhage"][i])
    # loading images
    img_path = Path("train", "image", str(i) + ".png")
    image_i = Image.open(img_path)
    data['data'].append(image_i)
    
# use np.unique to get all unique values in the list of labels
labels = np.unique(data['label'])
 
# set up the matplotlib figure and axes, based on the number of labels
fig, axes = plt.subplots(1, len(labels))
fig.set_size_inches(15,4)
fig.tight_layout()
 
# make a plot for every label (equipment) type. The index method returns the 
# index of the first item corresponding to its search string, label in this case
for ax, label in zip(axes, labels):
    idx = data['label'].index(label)
     
    ax.imshow(data['data'][idx], cmap = 'gray')
    ax.axis('off')
    ax.set_title(label)

hemo_count = 0
for i in data['label']:
    if i == 1:
        hemo_count += 1
print (hemo_count)