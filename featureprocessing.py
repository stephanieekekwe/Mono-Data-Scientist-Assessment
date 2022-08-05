import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

labelencoder_X = LabelEncoder()
feature_scaler = MinMaxScaler()


def feature_processing(transactions_data):
    # credit/debit
    encodede_transaction_type = labelencoder_X.fit_transform(transactions_data['type'])

    # narration
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=False)
    text_features = vectorizer.fit_transform(transactions_data['clean_narration'].values)

    #amount
    encoded_amount = np.array(transactions_data['amount']).reshape(-1, 1)

    x = feature_scaler.fit_transform(encoded_amount)

    train_data = pd.DataFrame(np.asarray(text_features.todense()),encodede_transaction_type)
    train_data['encoded_amount'] = x

    return train_data