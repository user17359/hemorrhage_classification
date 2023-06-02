import numpy as np
import pandas as pd
import train_test_split
from train_test_split import data, X_train, X_test, y_train, y_test, X, y
from HogTransformer import HogTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import svm

# create instances of transformers:
# (parameters after optimisation)

hog_images = HogTransformer(pixels_per_cell=(12, 12),
                            cells_per_block=(2, 2),
                            orientations=8,
                            block_norm='L2-Hys')
scale_images = MinMaxScaler()


# fit_transform method used to modify X_train - for transforming/scaling the data, the parameters (such as
# mean/variance) that the model has 'learnt' are used on the testing data later on
X_train_hog = hog_images.fit_transform(X_train)
X_train_final = scale_images.fit_transform(X_train_hog)

# print(X_train_final.shape)  # (2000, 60516)

# Stochastic Gradient Descent Classifier/Support Vector Machine Classifier
# create the classifier instance and train it:
svc = svm.SVC(C=0.1, kernel='linear', class_weight='balanced')

# class_weight = 'balanced' produced the same results as before (without using this parameter)
# changes appear only when the class weights are EXTREMELY imbalanced (I tried running the script with weights
# {0: 0.00001, 1: 200} - there were mistakes in the confusion matrix for the training set, accuracy dropped and
# the model was biased towards class 1 (more False Positives than before)
# so yeah I think there's not much we can do about it just by modifying the class weights

svc.fit(X_train_final, y_train)
y_train_pred = svc.predict(X_train_final)

labels = np.unique(y)

# classification metrics (training set):
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
y_test_pred = svc.predict(X_test_final)


# classification metrics (testing set):
print("Confusion matrix and classification report (testing data)\n")
cm2 = pd.DataFrame(confusion_matrix(y_test, y_test_pred), index=labels, columns=labels)
print(cm2)
print(classification_report(y_test, y_test_pred))
