# -*- coding: utf-8 -*-
"""
@author: Ahmad Qadri
Random Forest Classification on Heart Disease dataset

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#----  read data
heart_df = pd.read_csv('data\\heart2.csv')
rows, cols = heart_df.shape
print(f'> data rows = {rows}  data cols = {cols}')


#---- columns
heart_cols = heart_df.columns.tolist()
print(f'> columns = {heart_cols}')


#---- columns
heart_cols = heart_df.groupby('target').agg(testbps_mean=('trestbps', 'mean'), 
                                            thalach_mean=('thalach', 'mean')).round(1).reset_index()
print(heart_cols)


#---- plot THALACH with AGE
fig, ax = plt.subplots()
ax.scatter(heart_df['age'], heart_df['thalach'])
ax.set_title('THALACH vs AGE', fontsize=22, fontweight='bold')
ax.set_xlabel('AGE', fontsize=16, fontweight='bold')
ax.set_ylabel('THALACH', fontsize=16, fontweight='bold')
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.grid(True)
fig.savefig('eda/THALACH_vs_AGE.png')