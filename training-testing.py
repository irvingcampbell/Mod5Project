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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE


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
# Sample 1/4 of the data
df_ = df.groupby('FS').apply(lambda x: x.sample(frac = 0.25))
df_.index = df_.index.droplevel(0)
# Plot income distribution for SNAP participants and the rest
# of the population
%run visualizations.py
income_distribution_plot(df_)
# Drop geographical features and split the data into training and test
y = df_.FS
X = df_.drop(['FS', 'SERIALNO', 'REGION', 'DIVISION', 'ST', 'PUMA'], axis = 1)
# Split the data into test and training samples--stratify by SNAP recipiency
X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                    stratify = y, \
                                                    test_size = 0.25, \
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
y_train = y_train.astype('int')
sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train, y_train)
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
lr_clf.fit(X_train.drop(['GRNTP', 'NP'], axis = 1), y_train)
y_test_hat = lr_clf.predict(X_test.drop(['GRNTP', 'NP'], axis = 1))
print(classification_report(y_test, y_test_hat))
print(confusion_matrix(y_test, y_test_hat))

# Random forest classifier
rf_clf = RandomForestClassifier(random_state = 1007)
rf_clf.fit(X_train, y_train)
y_test_hat = rf_clf.predict(X_test)
print(classification_report(y_test, y_test_hat))
print(confusion_matrix(y_test, y_test_hat))


# AdaBoost
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 4), random_state = 1007)
ada_clf.fit(X_train, y_train)
y_test_hat = ada_clf.predict(X_test)
print(classification_report(y_test, y_test_hat))
print(confusion_matrix(y_test, y_test_hat))
ada_feat_importance = pd.DataFrame(zip(X_train.columns, \
                                       ada_clf.feature_importances_), \
                                   columns = ['Feature', 'Score'])
ada_feat_importance = ada_feat_importance.sort_values(by = 'Score', \
                                                      ascending = False)
ada_feat_importance.to_csv(f'{file_dir}ada_feat_importance.csv')

# Linear SVC 
svm_clf = LinearSVC(fit_intercept = 'False', random_state = 1007)
svm_clf.fit(X_train, y_train)
y_test_hat = svm_clf.predict(X_test)
confusion_matrix(y_test, y_test_hat)
print(classification_report(y_test, y_test_hat))
print(cm)

# Hyperparameter tuning
param_dist = {'C': [0.1, 10]}
gs = GridSearchCV(svm_clf, param_dist, cv = 3, scoring = 'recall')
gs.fit(X_train, y_train)
gs.best_params_

y_score = svm_clf.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label = 2)
print('AUC: {}'.format(auc(fpr, tpr)))
confusion_matrix_plot(cm)
roc_plot(fpr, tpr)
important_features_plot(X_train, svm_clf.coef_.ravel())

