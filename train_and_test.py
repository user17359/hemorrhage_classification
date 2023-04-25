import numpy as np
import pandas as pd
import train_test_split
from train_test_split import data, X_train, X_test, y_train, y_test, X, y
from HogTransformer import HogTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# create instances of transformers:
# (the parameters are only temporary, they need optimising)
hog_images = HogTransformer(pixels_per_cell=(12, 12),
                            cells_per_block=(2, 2),
                            orientations=9,
                            block_norm='L2-Hys')
scale_images = StandardScaler()

# fit_transform method used to modify X_train - for transforming/scaling the data, the parameters (such as
# mean/variance) that the model has 'learnt' are used on the testing data later on
X_train_hog = hog_images.fit_transform(X_train)
X_train_final = scale_images.fit_transform(X_train_hog)

# print(X_train_final.shape)  # (2000, 60516)

# Stochastic Gradient Descent Classifier

# create the classifier instance and train it:
sgd = SGDClassifier(random_state=321)
sgd.fit(X_train_final, y_train)
y_train_pred = sgd.predict(X_train_final)

labels = np.unique(y)

# classification metrics:
print("Confusion matrix and classification report (training data)\n")
cm = pd.DataFrame(confusion_matrix(y_train, y_train_pred), index=labels, columns=labels)
print(cm)
print("\n")
print(classification_report(y_train, y_train_pred))

# testing the classifier
# here, we only use the transform method, as we don't want the model to be biased
# (the model will use the same parameters as for the training data)
X_test_hog = hog_images.transform(X_test)
X_test_final = scale_images.transform(X_test_hog)
y_test_pred = sgd.predict(X_test_final)

# classification metrics:
print("Confusion matrix and classification report (testing data)\n")
cm2 = pd.DataFrame(confusion_matrix(y_test, y_test_pred), index=labels, columns=labels)
print(cm2)
print(classification_report(y_test, y_test_pred))
