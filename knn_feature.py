import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

from gokinjo import knn_kfold_extract
from gokinjo import knn_extract

def get_knn_feature(df_train, df_test):
    # label encode the target column
    le = LabelEncoder()
    df_train.target = le.fit_transform(df_train.target)

    # define X and y for training data
    X = df_train.drop(columns=["id","target"])
    y = df_train.target

    # prepare test data
    X_test=df_test.drop(columns="id")

    print("First five rows of training data:")
    display(X.head())
    print("First five rows of test data:")
    display(X_test.head())

    # convert to numpy because gokinjo expects np arrays
    X = X.to_numpy()
    y = y.to_numpy()
    X_test = X_test.to_numpy()
    # check shapes
    print("X, shape: ", np.shape(X))
    print("X_test, shape: ", np.shape(X_test))

    # KNN feature extraction for train, as the data has not been normalized previously, let knn_kfold_extract do it
    # you can set a different value for k, just be aware about the increase in computation time
    KNN_feat_train = knn_kfold_extract(X, y, k=5, normalize='standard')
    print("KNN features for training set, shape: ", np.shape(KNN_feat_train))
    KNN_feat_train[0]

    # create KNN features for test set, as the data has not been normalized previously, let knn_extract do it
    KNN_feat_test = knn_extract(X, y, X_test, k=5, normalize='standard')
    print("KNN features for test set, shape: ", np.shape(KNN_feat_test))
    KNN_feat_test[0]

    # add KNN feature to normal features
    X, X_test = np.append(X, KNN_feat_train, axis=1), np.append(X_test, KNN_feat_test, axis=1) 
    print("Train set, shape: ", np.shape(X))
    print("Test set, shape: ", np.shape(X_test))

    return KNN_feat_train, KNN_feat_test
