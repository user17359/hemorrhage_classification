import numpy as np
import pandas as pd
from train_test_split import data, X_train, X_test, y_train, y_test, X, y
from HogTransformer import HogTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm

# creating a pipeline for optimisation (hyper-parameters of the transformers & the classifier)
pipeline = Pipeline([('hog_images', HogTransformer(pixels_per_cell=(12, 12),
                                                   cells_per_block=(2, 2),
                                                   orientations=8,
                                                   block_norm='L2-Hys')),
                     ('scale_images', MinMaxScaler()),
                     ('classify', svm.SVC(kernel='linear'))])

# optimisation, run at your own risk :)))
# (takes a long time and requires plenty computing power)
# the hog transformer's parameters can still be optimised I think

params = [
    {
        'hog_images__pixels_per_cell': [(12, 12)],
        'hog_images__cells_per_block': [(2, 2)],
        'hog_images__orientations': [8, 9],
        'classify__C': [0.1, 0.5]
    }
]

clf = GridSearchCV(pipeline, params, scoring='accuracy', cv=3, verbose=2, n_jobs=-1)
clf.fit(X_train, y_train)

print('Best: ', clf.best_estimator_)
print('Best: ', clf.best_score_)
print('Best: ', clf.best_params_)