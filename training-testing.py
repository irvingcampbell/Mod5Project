#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures

y = df.FS
X = df.drop('FS', axis = 1)
# Split the data into test and training samples--stratify by year
X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                    stratify = y, \
                                                    test_size = 0.2, \
                                                    random_state = 1007)
