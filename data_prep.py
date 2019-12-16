#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: 
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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

def recode(data_frame):
    data_frame = data_frame.replace({'FS': {1: 2, 2: 1},
                                     'BATH': {1: 2, 2: 1},
                                     'REFR': {1: 2, 2: 1},
                                     'RNTM': {1: 2, 2: 1},
                                     'RWAT': {1: 2, 2: 1, 9:0},
                                     'RWATPR': {1: 2, 2: 1, 9:0},
                                     'SATELLITE': {1: 2, 2: 1},
                                     'SINK': {1: 2, 2: 1},
                                     'SMARTPHONE': {1: 2, 2: 1},
                                     'STOV': {1: 2, 2: 1},
                                     'TABLET': {1: 2, 2: 1},
                                     'TEL': {1: 2, 2: 1},
                                     'KIT': {1: 2, 2: 1},
                                     'PLM': {1: 2, 2: 1, 9: 0},
                                     'PLMPRP': {1: 2, 2: 1, 9: 0},
                                     'SRNT': {0: 2},
                                     'SVAL': {0: 2},
                                     'TEN': {1: 3, 2: 4, 3: 2, 4: 1},
                                     'ELEFP': {1: 2, 2: 1},
                                     'FULFP': {1: 2, 2: 1},
                                     'GASFP': {1: 2, 2: 1}})
    return data_frame
    
# Imputer
def imputer(data_frame):
    data_frame = data_frame.drop(data_frame.FS.isna(), \
                                 axis = 0)
    data_frame = data_frame.drop(data_frame.HINCP.isna(), \
                                 axis = 0)
    data_frame['HINCP'] = data_frame.HINCP * \
                          data_frame.ADJINC / (10**6) 
    imptr = SimpleImputer(strategy = 'constant', fill_value = 99)
    data_frame['WIF'] = imptr(data_frame.WIF)
    data_frame['WORKSTAT'] = imptr(data_frame.WORKSTAT)
    imptr = SimpleImputer(strategy = 'constant', fill_value = 0)
    data_frame['VEH'] = imptr(data_frame.VEH)
    data_frame['BATH'] = imptr(data_frame.BATH)
    data_frame['REFR'] = imptr(data_frame.REFR)
    data_frame['SINK'] = imptr(data_frame.SINK)
    data_frame['STOV'] = imptr(data_frame.STOV)
    data_frame['KIT'] = imptr(data_frame.KIT)
    data_frame['BDSP'] = imptr(data_frame.BDSP)
    data_frame['BLD'] = imptr(data_frame.BLD)
    data_frame['RWAT'] = imptr(data_frame.RWAT)
    data_frame['RWATPR'] = imptr(data_frame.RWATPR)
    data_frame['PLM'] = imptr(data_frame.PLM)
    data_frame['PLMPR'] = imptr(data_frame.PLMPR)
    data_frame['SRNT'] = imptr(data_frame.SRNT)
    data_frame['SVAL'] = imptr(data_frame.SVAL)
    data_frame['MV'] = imptr(data_frame.MV)
    data_frame['TEN'] = imptr(data_frame.TEN)
    data_frame['CONP'] = imptr(data_frame.CONP)
    data_frame['ELEFP'] = imptr(data_frame.ELEFP)
    data_frame['ELEP'] = imptr(data_frame.ELEP)
    data_frame['FULFP'] = imptr(data_frame.FULFP)
    data_frame['FULP'] = imptr(data_frame.FULP)
    data_frame['GASFP'] = imptr(data_frame.GASFP)
    data_frame['GASP'] = imptr(data_frame.GASP)
    data_frame['HFL'] = imptr(data_frame.HFL)
    data_frame = data_frame.drop(['ACR', 'AGS', 'FINCP', 'VALP', 'VACS'], axis = 1)
    
    
# Import the nationwide data-it is presented in two files 
file_dir = '/Users/flatironschol/FIS-Projects/Module5/data/'
df_a = pd.read_csv(f'{file_dir}psam_husa.csv')
df_b = pd.read_csv(f'{file_dir}psam_husb.csv')
df_a = drop_weights(df_a)
df_a = drop_allocations(df_a)
df_b = drop_weights(df_b)
df_b = drop_allocations(df_b)
df = pd.concat((df_a, df_b), axis = 0) 
y = df.FS
X = df.drop('FS', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, \
                                                    stratify = y, random_state = 1007)