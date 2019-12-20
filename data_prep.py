#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is for FIS Module 5 Project.
In this project, we examine the efficiency and effectiveness of 
SNAP, the nation's anti-hunger program.
Specifically, the code cleans the raw ACS data (2018) for the
entire US.
"""
import pandas as pd
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

# Function to recode the categorical features 
def recode(data_frame):
    # Binary categories
    dict_bin = {0: 1, 1: 2}
    bin_cols = ['HUGCL', 'NPP', 'NR', 'PSF', 'R18']
    for c in bin_cols:
        data_frame[c] = data_frame[c].map(dict_bin)
    dict_bin = {1: 2, 2: 1}
    bin_cols = ['BROADBND', 'COMPOTHX', 'DIALUP', 'HISPEED', 'LAPTOP', 'OTHSVCEX', \
                'SATELLITE', 'SMARTPHONE', 'TABLET', 'TEL', 'FS', 'BATH', 'REFR', \
                'SINK', 'STOV', 'KIT', 'RNTM']
    for c in bin_cols:
        data_frame[c] = data_frame[c].map(dict_bin)
    dict_bin = {0: 2, 1: 1}
    bin_cols = ['SRNT', 'SVAL']
    for c in bin_cols:
        data_frame[c] = data_frame[c].map(dict_bin)
    # Three categories
    dict_bin = {1: 2, 2: 1, 9:0}
    bin_cols = ['RWAT', 'RWATPR', 'PLM', 'PLMPRP', 'HOTWAT']
    for c in bin_cols:
        data_frame[c] = data_frame[c].map(dict_bin)
    dict_bin = {1: 2, 2: 1, 3: 3}
    bin_cols = ['ELEFP', 'FULFP', 'GASFP', 'WATFP']
    for c in bin_cols:
        data_frame[c] = data_frame[c].map(dict_bin)
    return data_frame
    
# Customized imputer function
# Since fill_value is a constant, we apply the function to the entire dataset
def imputer(data_frame):
    imptr = SimpleImputer(strategy = 'constant', fill_value = 0)
    relevant_cols = ['SERIALNO', 'REGION', 'DIVISION', 'ST', 'PUMA', 'FS', \
                     'HINCP', 'ADJINC', 'WIF', 'WORKSTAT', 'VEH', 'NP', 'HHL', \
                     'FPARC', 'HHT', 'HUGCL', 'HUPAC', 'LNGI', 'MULTG', 'NPF', \
                     'NPP', 'NR', 'NRC', 'PARTNER', 'PSF', 'R18', 'R65', 'SSMC', \
                     'ACCESS', 'BROADBND', 'COMPOTHX', 'DIALUP', 'HISPEED', 'LAPTOP', \
                     'OTHSVCEX', 'SATELLITE', 'SMARTPHONE', 'TABLET', 'TEL', \
                     'TYPE', 'BATH', 'BDSP', 'BLD', 'REFR', 'RMSP', 'RWAT', \
                     'RWATPR', 'SINK', 'STOV', 'TEN', 'KIT', 'MV', 'PLM', 'PLMPRP', \
                     'HOTWAT', 'SRNT', 'SVAL', 'YBL', 'CONP', 'ELEFP', 'ELEP', \
                     'FULFP', 'FULP', 'GASFP', 'GASP', 'HFL', 'INSP', 'MHP', \
                     'RNTM', 'RNTP', 'WATFP', 'WATP', 'GRNTP', 'SMOCP']
    data_frame = data_frame[relevant_cols]
    data_frame = data_frame.dropna(subset = ['FS', 'HINCP'])    
    data_frame = imptr.fit_transform(data_frame)
    return pd.DataFrame(data_frame, columns = relevant_cols)
    
# Import the nationwide data-it is presented in two files 
file_dir = '/Users/flatironschol/FIS-Projects/Module5/data/'
df_a = pd.read_csv(f'{file_dir}psam_husa.csv')
df_b = pd.read_csv(f'{file_dir}psam_husb.csv')
df_a = drop_weights(df_a)
df_a = drop_allocations(df_a)
df_b = drop_weights(df_b)
df_b = drop_allocations(df_b)
df = pd.concat((df_a, df_b), axis = 0) 
df = recode(df)
df = imputer(df)
df.to_csv(f'{file_dir}df.csv')