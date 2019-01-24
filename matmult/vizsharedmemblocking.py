# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 10:04:42 2019

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

df1 = pd.read_csv('./data/sharedmem_blocksize.dat',delim_whitespace=True,header=None,
                  names=['Block Size','Memory','MFlops','hash','function','Size']).drop(['hash','function'],axis=1)
df_comb = df1

plt.figure()
df_comb.set_index("Block Size", inplace=True)
ax = df_comb.groupby("Size")["MFlops"].plot(legend=True, style ='*-')
plt.legend(loc='upper left')
plt.xlabel('Block Size')
plt.ylabel('MFlops')
plt.gca().set_ylim(bottom=0)
plt.xscale('log',basex=2)
#plt.yscale('log',basey=2)
ax = plt.gca().xaxis 
ax.set_major_formatter(ScalarFormatter()) 
#ax = plt.gca().yaxis 
#ax.set_major_formatter(ScalarFormatter()) 
plt.savefig('BlockingSize.png', bbox_inches='tight')
plt.close()
