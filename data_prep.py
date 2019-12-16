#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: 
"""
import numpy as np
import pandas as pd
import sklearn.model_selection as train_test_split

# Function to drop weight columns
def drop_weights(data_frame):
    coltodrop = []
    for i in range(1, 81):
        coltodrop.append('WGTP' + str(i))
    return data_frame.drop(coltodrop, axis = 1)

# Function to drop allocation columns
def drop_allocations(data_frame):
    col = data_frame.columns
    coltodrop = []
    for c in col:
        if (c[0]== 'F') & (c[-1] == 'P') & \
           (c != 'FULFP') & (c != 'FULP') & (c != 'FINCP'): 
            coltodrop.append(c)
    return data_frame.drop(coltodrop, axis = 1)

# Imputer
def imputer(data_frame):
    data_frame = data_frame.drop(data_frame.FS.isna(), \
                                 axis = 0)
    data_frame = data_frame.drop(data_frame.HINCP.isna(), \
                                 axis = 0)
    data_frame['HINCP'] = data_frame.HINCP * \
                          data_frame.ADJINC / (10**6) 
    data_frame = data_frame.drop('FINCP', axis = 1)
    
    
# Import the nationwide data-it is presented in two files 
file_dir = '/Users/flatironschol/FIS-Projects/Module5/data/'
df_a = pd.read_csv(f'{file_dir}psam_husa.csv')
df_b = pd.read_csv(f'{file_dir}psam_husb.csv')
df_a = drop_weights(df_a)
df_a = drop_allocations(df_a)
df_b = drop_weights(df_b)
df_b = drop_allocations(df_b)
df = pd.concat((df_a, df_b), axis = 0) 
