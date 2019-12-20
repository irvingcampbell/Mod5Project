#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is for FIS Module 5 Project.
In this project, we examine the efficiency and effectiveness of 
SNAP, the nation's anti-hunger program.
Specifically, the code provides descriptive visualizations.
"""
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")
sns.set_style("white")

# Plot income distribution for SNAP beneficiaries vis-a-vis
# the rest of the population
def income_distribution_plot(df_):
    bin_snap = df_.loc[df_.FS == 2].HINCP.max() / 2000
    bin_rest = df_.loc[df_.FS == 1].HINCP.max() / 2000
    fig, ax1 = plt.subplots(1, 1, figsize = (10, 6))
    sns.distplot(df_.loc[df_.FS == 2].HINCP, bins = int(bin_snap), ax = ax1, \
                 kde = False, norm_hist = True, label = 'SNAP recipient', \
                 color = 'blue')
    sns.distplot(df_.HINCP, bins = int(bin_rest), ax = ax1, \
                 kde = False, norm_hist = True, label = 'Entire population', \
                 color = 'purple')
    ax1.set(yticklabels=[])
    ax1.set_xlim(0, 300000)
    ax1.set_xlabel('Annual household income, US$')
    ax1.set_title('Income Distribution of SNAP Recipients, 2018')
    ax1.legend()
    plt.show()
    return

# Plot ROC
def roc_plot(fpr, tpr):
    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)], rotation = 'vertical')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    return

# Plot confusion matrix
def confusion_matrix_plot(cm):
    # create normalized confusion matrix <cm_nor>
    cm_nor = np.zeros((cm.shape[0], cm.shape[1]))
    for col in range(cm.shape[1]):
        cm_nor[:, col] = (cm[:, col] / sum(cm[:, col]))
        plt.ylim(-10, 10)
    # create normalized confusion matrix heat map
    sns.heatmap(cm_nor, cmap="Blues", annot=True, annot_kws={"size": 14})
    locs, labels = plt.xticks()
    plt.xticks(locs, ("NO", "YES"))
    locs, labels = plt.yticks()
    plt.yticks(locs, ("NO", "YES"))
    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("SNAP Enrollment Predictions")
    plt.show()
    return

# Plot important features for LinearSVC
def important_features_plot(coef):
    top_features = 5
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['purple' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(X_train.columns)
    real_top_feat_names = feature_names[top_coefficients]
    x_labels = ['Income', 'Same-sex couple', 'Rent', 'Age of children', \
                'Number of children', 'Meals included in rent', 'Not same-sex couple', \
                'Household size', 'Unemployment', 'Family size']
    plt.xticks(np.arange(0, 1 + 2 * top_features), x_labels, rotation=90)
    plt.title('Characteristics to Identify SNAP Beneficiaries')
    plt.show()
    return