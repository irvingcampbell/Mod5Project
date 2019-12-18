#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Function to calculate VIF scores
def calculate_vif(X, thresh=8):
    feats = X.columns
    while len(feats) >= 2:
        vif = [variance_inflation_factor(X[feats].values, i) for i in range(len(feats))]
        if max(vif) > thresh:
            maxloc = vif.index(max(vif))
            print('dropping \'' + feats[maxloc] + '\' at index: ' + str(maxloc))
            feats.remove(feats[maxloc])
    print('Remaining variables:')
    print(feats)
    return X[feats]

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

file_dir = '/Users/flatironschol/FIS-Projects/Module5/data/'
df = pd.read_csv(f'{file_dir}df.csv', index_col = 0)
# df_ = df.groupby('FS').apply(lambda x: x.sample(frac = 0.25))
# df_.index = df_.index.droplevel(0)        
# Focus only on California
df_ = df.loc[df.ST == 6]
y = df_.FS
X = df_.drop(['FS', 'SERIALNO', 'REGION', 'DIVISION', 'ST', \
              'HOTWAT', 'RWATPR', 'PLMPRP'], axis = 1)
# Split the data into test and training samples--stratify by SNAP recipiency
X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                    stratify = y, \
                                                    test_size = 0.25, \
                                                    random_state = 1007)
sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train, y_train)
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
y_train = y_train.astype('int')

# X_train.to_csv(f'{file_dir}X_train.csv')
# y_train.to_csv(f'{file_dir}y_train.csv')

# Scaling and one-hot-encoding of the test set
X_test_cont = pd.DataFrame(sclr.transform(X_test[cont_feats]), \
                           columns = cont_feats)
X_test_cat = encoder_transform(encdr, X_test[cat_feats])
X_test = pd.concat((X_test_cont, X_test_cat), axis = 1)
y_test = y_test.astype('int')
# X_test.to_csv(f'{file_dir}X_test.csv')
# y_test.to_csv(f'{file_dir}y_test.csv')

# Logistical regression
lr_clf = LogisticRegression(random_state=1007, solver='saga')
lr_clf.fit(X_train, y_train)
y_train_hat = lr_clf.predict(X_train)
print(classification_report(y_train, y_train_hat))
print(confusion_matrix(y_train, y_train_hat))
y_test_hat = lr_clf.predict(X_test)
print(classification_report(y_test, y_test_hat))
print(confusion_matrix(y_test, y_test_hat))

# Random forest classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
y_train_hat = rf_clf.predict(X_train)
print(classification_report(y_train, y_train_hat))
print(confusion_matrix(y_train, y_train_hat))
y_test_hat = rf_clf.predict(X_test)
print(classification_report(y_test, y_test_hat))
print(confusion_matrix(y_test, y_test_hat))

# Gridsearch for rf_clf
params = {'n_estimators': [10, 100, 200],
          'max_depth': [5, 10, None],
          'min_samples_split': [2, 5, 10],
          'min_samples_leaf': [1, 5, 10],
          'max_features': ['auto', None]}
gs = GridSearchCV(rf_clf, params, scoring = 'recall', cv = 3, n_jobs = -1)
gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

# Linear SVC 
svm_clf = LinearSVC()
svm_clf.fit(X_train.values, y_train.values)
y_train_hat = svm_clf.predict(X_train.values)
print(classification_report(y_train, y_train_hat))
print(confusion_matrix(y_train, y_train_hat))
y_test_hat = svm_clf.predict(X_test.values)
print(classification_report(y_test, y_test_hat))
print(confusion_matrix(y_test, y_test_hat))

# AdaBoost
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 4))
ada_clf.fit(X_train_cont, y_train)
y_train_hat = ada_clf.predict(X_train_cont)
print(classification_report(y_train, y_train_hat))
print(confusion_matrix(y_train, y_train_hat))
y_test_hat = ada_clf.predict(X_test_cont)
print(classification_report(y_test, y_test_hat))
print(confusion_matrix(y_test, y_test_hat))