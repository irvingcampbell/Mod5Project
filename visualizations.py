#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker

sns.set_context("talk")
sns.set_style("white")

fig, ax1 = plt.subplots(1, 1, figsize = (10, 6))
sns.distplot(df.loc[df.FS == 2].HINCP, bins = 500, ax = ax1, \
             kde = False, norm_hist = True, label = 'SNAP recipient', \
             color = 'blue')
sns.distplot(df.HINCP, bins = 500, ax = ax1, \
             kde = False, norm_hist = True, label = 'Entire population', \
             color = 'purple')
ax1.set_xlim(0, 300000)
ax1.set_xlabel('Annual household income, US$')
ax1.set_title('Income Distribution of SNAP Recipients, 2018')
ax1.legend()
