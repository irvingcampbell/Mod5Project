#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")
sns.set_style("white")

fig, ax1 = plt.subplots(1, 1, figsize = (10, 6))
bin_snap = df_.loc[df_.FS == 2].HINCP.max() / 5000
bin_rest = df_.loc[df_.FS == 1].HINCP.max() / 5000
sns.distplot(df_.loc[df_.FS == 2].HINCP, bins = 500, ax = ax1, \
             kde = False, norm_hist = True, label = 'SNAP recipient', \
             color = 'blue')
sns.distplot(df_.HINCP, bins = 500, ax = ax1, \
             kde = False, norm_hist = True, label = 'Entire population', \
             color = 'purple')
ax1.set_xlim(0, 300000)
ax1.set_xlabel('Annual household income, US$')
ax1.set_title('Income Distribution of SNAP Recipients, 2018')
ax1.legend()

sns.barplot(y = 'Feature', x = 'Score', data = ada_feat_importance[0:5], \
            color = 'b', orient = 'h')

y_test_hat = ada_clf.predict(X_test)
labels = ['non-SNAP', 'SNAP']
cm = confusion_matrix(y_test, y_test_hat)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()