import numpy as np
import pandas as pd
import train_test_split
from train_test_split import data, X_train, X_test, y_train, y_test, X, y
from HogTransformer import HogTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import svm

# create instances of transformers:
# (the parameters are only temporary, they need optimising)

hog_images = HogTransformer(pixels_per_cell=(12, 12),
                            cells_per_block=(2, 2),
                            orientations=9,
                            block_norm='L2-Hys')
scale_images = StandardScaler()

'''
# creating a pipeline for optimisation (hyper-parameters of the transformers & the classifier)
pipeline = Pipeline([('hog_images', HogTransformer(pixels_per_cell=(12, 12),
                                                   cells_per_block=(2, 2),
                                                   orientations=8,
                                                   block_norm='L2-Hys')),
                     ('scale_images', StandardScaler()),
                     ('classify', svm.SVC(kernel='linear'))])

optimisation, run at your own risk :)))
(takes a long time and requires plenty computing power)
the hog transformer's parameters can still be optimised I think

params = [
    {
        'hog_images__pixels_per_cell': [(8, 8), (9, 9), (12, 12)],
        'hog_images__cells_per_block': [(2, 2)],
        'hog_images__orientations': [8],
        'classify': [SGDClassifier(random_state=321), svm.SVC(kernel='linear')]
    }
]

clf = GridSearchCV(pipeline, params, scoring='accuracy', cv=3, verbose=2, n_jobs=-1)
clf.fit(X_train, y_train)

print('Best: ', clf.best_estimator_)
print('Best: ', clf.best_score_)
print('Best: ', clf.best_params_)
'''
# fit_transform method used to modify X_train - for transforming/scaling the data, the parameters (such as
# mean/variance) that the model has 'learnt' are used on the testing data later on
X_train_hog = hog_images.fit_transform(X_train)
X_train_final = scale_images.fit_transform(X_train_hog)

# print(X_train_final.shape)  # (2000, 60516)

# Stochastic Gradient Descent Classifier/Support Vector Machine Classifier
# create the classifier instance and train it:
svc = svm.SVC(kernel='linear', class_weight='balanced')

# class_weight = 'balanced' produced the same results as before (without using this parameter)
# changes appear only when the class weights are EXTREMELY imbalanced (I tried running the script with weights
# {0: 0.00001, 1: 200} - there were mistakes in the confusion matrix for the training set, accuracy dropped and
# the model was biased towards class 1 (more False Positives than before)
# so yeah I think there's not much we can do about it just by modifying the class weights, we should try data
# augmentation as our next step

svc.fit(X_train_final, y_train)
y_train_pred = svc.predict(X_train_final)

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
y_test_pred = svc.predict(X_test_final)


# classification metrics:
print("Confusion matrix and classification report (testing data)\n")
cm2 = pd.DataFrame(confusion_matrix(y_test, y_test_pred), index=labels, columns=labels)
print(cm2)
print(classification_report(y_test, y_test_pred))
