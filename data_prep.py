#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: 
"""
import numpy as np
import pandas as pd

file_dir = '/Users/flatironschol/FIS-Projects/Module5/data/'
df_p = pd.read_csv(f'{file_dir}psam_h11.csv')
df_p.FS.value_counts()