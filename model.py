from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


def model(train_data, transactions_data):
    model = KMeans(n_clusters=15, init='k-means++',
                   n_init=50, random_state=5,  max_iter=50)
    model.fit_predict(train_data)

    transactions_data['group'] = model.labels_
    return transactions_data
