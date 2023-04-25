from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

labels = pd.read_csv("hemorrhage_labels.csv")

# creating dict for data
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
fig.suptitle("Example images")
 
# make a plot for every label (equipment) type. The index method returns the 
# index of the first item corresponding to its search string, label in this case
for ax, label in zip(axes, labels):
    idx = data['label'].index(label)
     
    ax.imshow(data['data'][idx], cmap = 'gray')
    ax.axis('off')
    ax.set_title(label)
plt.show()

# code for testing dictionary
"""hemo_count = 0
for i in data['label']:
    if i == 1:
        hemo_count += 1
print (hemo_count)"""

# splitting the data into training and test sets
X = data['data']
y = np.array(data['label'])
 
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    shuffle=True,
    random_state=42,
    stratify=y
)

# checking if sets are equal in size
unique, train_counts = np.unique(y_train, return_counts=True)
unique, test_counts = np.unique(y_train, return_counts=True)

y_train_0 = train_counts[0] / np.sum(train_counts) * 100
y_train_1 = train_counts[1] / np.sum(train_counts) * 100

y_test_0 = test_counts[0] / np.sum(test_counts) * 100
y_test_1 = test_counts[1] / np.sum(test_counts) * 100

X = ['No hemorrhage', 'Hemorrhage']
Ytrain = [y_train_0, y_train_1]
Ytest = [y_test_0, y_test_1]
  
X_axis = np.arange(len(X))

# histogram
plt.bar(X_axis - 0.2, Ytrain, 0.4, label = 'Train set', color = "pink")
plt.bar(X_axis + 0.2, Ytest, 0.4, label = 'Test set', color = "purple")
  
plt.xticks(X_axis, X)
plt.xlabel("Classes")
plt.ylabel("Percentage of images")
plt.title("Percentage of classes in each set")
plt.legend()
plt.show()
