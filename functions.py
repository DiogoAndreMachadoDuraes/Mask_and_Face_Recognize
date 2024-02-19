import numpy as np
import cv2 as cv
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import warnings


def load_data():

    data = {
        "archive": [],
        "label": [],
        "target": [],
        "img": [],
    }

    with_mask = os.listdir("img/maskon")
    without_mask = os.listdir("img/maskoff")

    for archive in with_mask:
        data["archive"].append(f"img/maskon/{archive}")
        data["label"].append(f"With mask")
        data["target"].append(1)
        img = cv.cvtColor(cv.imread(f"img/maskon/{archive}"), cv.COLOR_BGR2GRAY).flatten()
        data["img"].append(img)
        
    for archive in without_mask:
        data["archive"].append(f"img/maskoff/{archive}")
        data["label"].append(f"Without mask")
        data["target"].append(0)
        img = cv.cvtColor(cv.imread(f"img/maskoff/{archive}"), cv.COLOR_BGR2GRAY).flatten()
        data["img"].append(img)

    data_frame = pd.DataFrame(data)

    return data_frame

def train_test(dataframe):

    X = list(dataframe["img"])
    y = list(dataframe["target"])

    return train_test_split(X, y, train_size=0.40, random_state=13)

def pca_model(X_train):

    pca = PCA(n_components=30)
    pca.fit(X_train)
    
    return pca

def knn(X_train, y_train):

    warnings.filterwarnings("ignore")

    grid_params = {
    "n_neighbors": [2, 3, 5, 11, 19, 23, 29],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattam", "cosine", "l1", "l2"]
    }
    
    knn_model = GridSearchCV(KNeighborsClassifier(), grid_params, refit=True)

    knn_model.fit(X_train, y_train)

    return knn_model