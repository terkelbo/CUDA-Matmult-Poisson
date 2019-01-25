# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 09:30:23 2019

@author: terkelbo-pc
"""

import sys
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

font = {'size'   : 14}

mpl.rc('font', **font)

df1 = pd.read_csv('./data/registerblocking.dat',delim_whitespace=True,header=None,
                  names=['Register Size','Memory','MFlops','hash','function']).drop(['hash','function'],axis=1)
df2 = pd.read_csv('./data/registerblocking_down.dat',delim_whitespace=True,header=None,
                  names=['Register Size','Memory','MFlops','hash','function']).drop(['hash','function'],axis=1)

df1['Blocking Direction'] = 'Along rows of C'
df2['Blocking Direction'] = 'Along columns of C'

df_comb = pd.concat((df1,df2))
df_comb['GFlops'] = df_comb['MFlops']/1000

plt.figure()
df_comb.set_index("Register Size", inplace=True)
ax = df_comb.groupby("Blocking Direction")["GFlops"].plot(legend=True, 
style ='*-')
plt.legend(loc='upper right')
plt.xlabel('# of elements per thread')
plt.ylabel('GFlop/s')
plt.gca().set_ylim(bottom=0)
plt.xscale('log',basex=2)
#plt.yscale('log',basey=2)
ax = plt.gca().xaxis 
ax.set_major_formatter(ScalarFormatter()) 
#ax = plt.gca().yaxis 
#ax.set_major_formatter(ScalarFormatter()) 
plt.savefig('RegisterSize.png', bbox_inches='tight')
plt.close()
