import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import sys
import shutil
import os

                     
                     
df1 = pd.read_csv('statfun_gpu1.dat',delim_whitespace=True,header=None,
	names=['memory','Mflop/s', 'dummy', 'implementation', 'rows'])\
	.drop(['dummy','implementation', 'rows'],axis=1)
df2 = pd.read_csv('statfun_gpu2.dat',delim_whitespace=True,header=None,
	names=['memory','Mflop/s', 'dummy', 'implementation', 'rows'])\
    .drop(['dummy','implementation', 'rows'],axis=1)
df3 = pd.read_csv('statfun_gpu3.dat',delim_whitespace=True,header=None,
	names=['memory','Mflop/s', 'dummy', 'implementation', 'rows'])\
	.drop(['dummy','implementation', 'rows'],axis=1)
df4 = pd.read_csv('statfun_gpu4.dat',delim_whitespace=True,header=None,
	names=['memory','Mflop/s', 'dummy', 'implementation', 'rows'])\
	.drop(['dummy','implementation', 'rows'],axis=1)
df5 = pd.read_csv('statfun_gpu5.dat',delim_whitespace=True,header=None,
	names=['memory','Mflop/s', 'dummy', 'implementation', 'rows'])\
	.drop(['dummy','implementation', 'rows'],axis=1)
'''
df6 = pd.read_csv('statfun_lib.dat',delim_whitespace=True,header=None,
	names=['memory','Mflop/s', 'dummy', 'implementation', 'rows'])\
	.drop(['dummy','implementation', 'rows'],axis=1)
df7 = pd.read_csv('statfun_gpulib.dat',delim_whitespace=True,header=None,
	names=['memory','Mflop/s', 'dummy', 'implementation', 'rows'])\
	.drop(['dummy','implementation', 'rows'],axis=1)
'''        
                    																
plt.figure()
ax = df1.plot('memory',logx=False, style='.-')
df2.plot('memory', ax=ax, style='*-')
df3.plot('memory', ax=ax, style='*-')
df4.plot('memory', ax=ax, style='*-')
df5.plot('memory', ax=ax, style='*-')
#df6.plot('memory', ax=ax, style='*-')
#df7.plot('memory', ax=ax, style='*-')
plt.legend(loc='upper left')
plt.xlabel('Memory Footprint (Kbyte)')
plt.ylabel('Performance (Mflops/s)')
plt.gca().set_ylim(bottom=0)
plt.xscale('log',basex=4)
ax = plt.gca().xaxis 
ax.set_major_formatter(ScalarFormatter()) 
plt.savefig('Mflops_matmul.png', bbox_inches='tight')
plt.close()
	                     																		
                     																		
