#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Function to calculate VIF scores and drop the 


# Function to transform categorical variables
    
def encoder_transform(encoder, X):
    X_encoded = encoder.transform(X).toarray()
    encoded_feats = list(encoder.get_feature_names())
    feats = X.columns
    encoded_feats_updated = []
    for feat in encoded_feats:
        feat_split = feat.split('_')
        i = int(feat_split[0][1:])
        dummies = feat_split[1]
        feat_updated = f'{feats[i]}_{dummies}'
        encoded_feats_updated.append(feat_updated)
    return pd.DataFrame(X_encoded, columns = encoded_feats_updated)
        
y = df.FS
X = df.drop(['FS', 'SERIALNO'], axis = 1)
# Split the data into test and training samples--stratify by SNAP recipiency
X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                    stratify = y, \
                                                    test_size = 0.2, \
                                                    random_state = 1007)
# Scaling and one-hot-encoding of the training set
cont_feats = ['HINCP', 'VEH', 'NP', 'NPF', 'NRC', 'BDSP', 'BLD', 'RMSP', \
              'YBL', 'CONP', 'ELEP', 'GASP', 'FULP', 'INSP', 'MHP', \
              'RNTP', 'WATP', 'GRNTP', 'SMOCP']
sclr = StandardScaler()
X_train_cont = pd.DataFrame(sclr.fit_transform(X_train[cont_feats]), \
                            columns = cont_feats)
cat_feats = X_train.drop(cont_feats, axis = 1).columns
encdr = OneHotEncoder(handle_unknown = 'ignore')
encdr.fit(X_train[cat_feats])
X_train_cat = encoder_transform(encdr, X_train[cat_feats])
X_train = pd.concat((X_train_cont, X_train_cat), axis = 1)

# Scaling and one-hot-encoding of the test set
X_test_cont = pd.DataFrame(sclr.transform(X_test[cont_feats]), \
                           columns = cont_feats)
X_test_cat = encoder_transform(encdr, X_test[cat_feats])
X_test = pd.concat((X_test_cont, X_test_cat), axis = 1)