import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
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
df6 = pd.read_csv('statfun_lib.dat',delim_whitespace=True,header=None,
	names=['memory','Mflop/s', 'dummy', 'implementation', 'rows'])\
	.drop(['dummy','implementation', 'rows'],axis=1)
df7 = pd.read_csv('statfun_gpulib.dat',delim_whitespace=True,header=None,
	names=['memory','Mflop/s', 'dummy', 'implementation', 'rows'])\
	.drop(['dummy','implementation', 'rows'],axis=1)

df1['Mflop/s'] = df1['Mflop/s']/1000 
df2['Mflop/s'] = df2['Mflop/s']/1000
df3['Mflop/s'] = df3['Mflop/s']/1000
df4['Mflop/s'] = df4['Mflop/s']/1000
df5['Mflop/s'] = df5['Mflop/s']/1000
df6['Mflop/s'] = df6['Mflop/s']/1000
df7['Mflop/s'] = df7['Mflop/s']/1000       
df1['memory'] = df1['memory']/1000
df2['memory'] = df2['memory']/1000
df3['memory'] = df3['memory']/1000
df4['memory'] = df4['memory']/1000
df5['memory'] = df5['memory']/1000
df6['memory'] = df6['memory']/1000
df7['memory'] = df7['memory']/1000
                    																
plt.figure()
ax = df1.plot('memory',logx=False, style='.-')
df2.plot('memory', ax=ax, style='*-')
df3.plot('memory', ax=ax, style='*-')
df4.plot('memory', ax=ax, style='*-')
df5.plot('memory', ax=ax, style='*-')
df6.plot('memory', ax=ax, style='*-')
df7.plot('memory', ax=ax, style='*-')
plt.legend(["GPU1","GPU2","GPU3","GPU4","GPU5",'CPU','cublasDgemm'],loc='lower right')
plt.xlabel('Memory Footprint (Mbyte)')
plt.ylabel('Performance (Gflops/s)')
#plt.gca().set_ylim(bottom=0)
plt.xscale('log',basex=2)
plt.yscale('log',basey=2)
ax = plt.gca().xaxis
ax.set_major_formatter(FormatStrFormatter('%.2f'))
ax = plt.gca().yaxis
ax.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(rotation=45)
plt.savefig('Mflops_matmul.png', bbox_inches='tight')
plt.close()
	                     																		
                     																		
plt.figure()
ax = df1.plot('memory',logx=False, style='.-')
df6.plot('memory', ax=ax, style='*-')
plt.legend(["GPU1","CPU"],loc='upper left')
plt.xlabel('Memory Footprint (Mbyte)')
plt.ylabel('Performance (Gflops/s)')
#plt.gca().set_ylim(bottom=0)
plt.xscale('log',basex=2)
plt.yscale('log',basey=2)
ax = plt.gca().xaxis
ax.set_major_formatter(FormatStrFormatter('%.2f'))
ax = plt.gca().yaxis
ax.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(rotation=45)
plt.savefig('Mflops_GPU1_matmul.png', bbox_inches='tight')
plt.close()


plt.figure()
ax = df2.plot('memory',logx=False, style='.-')
df6.plot('memory', ax=ax, style='*-')
plt.legend(["GPU2","CPU"],loc='upper left')
plt.xlabel('Memory Footprint (Mbyte)')
plt.ylabel('Performance (Gflops/s)')
#plt.gca().set_ylim(bottom=0)
plt.xscale('log',basex=2)
plt.yscale('log',basey=2)
ax = plt.gca().xaxis
ax.set_major_formatter(FormatStrFormatter('%.2f'))
ax = plt.gca().yaxis
ax.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(rotation=45)
plt.savefig('Mflops_GPU2_matmul.png', bbox_inches='tight')
plt.close()

plt.figure()
ax = df3.plot('memory',logx=False, style='.-')
df6.plot('memory', ax=ax, style='*-')
plt.legend(["GPU1","CPU"],loc='upper left')
plt.xlabel('Memory Footprint (Mbyte)')
plt.ylabel('Performance (Gflops/s)')
#plt.gca().set_ylim(bottom=0)
plt.xscale('log',basex=2)
plt.yscale('log',basey=2)
ax = plt.gca().xaxis
ax.set_major_formatter(FormatStrFormatter('%.2f'))
ax = plt.gca().yaxis
ax.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(rotation=45)
plt.savefig('Mflops_GPU3_matmul.png', bbox_inches='tight')
plt.close()

plt.figure()
ax = df4.plot('memory',logx=False, style='.-')
df6.plot('memory', ax=ax, style='*-')
plt.legend(["GPU4","CPU"],loc='upper left')
plt.xlabel('Memory Footprint (Mbyte)')
plt.ylabel('Performance (Gflops/s)')
#plt.gca().set_ylim(bottom=0)
plt.xscale('log',basex=2)
plt.yscale('log',basey=2)
ax = plt.gca().xaxis
ax.set_major_formatter(FormatStrFormatter('%.2f'))
ax = plt.gca().yaxis
ax.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(rotation=45)
plt.savefig('Mflops_GPU4_matmul.png', bbox_inches='tight')
plt.close()

plt.figure()
ax = df5.plot('memory',logx=False, style='.-')
df6.plot('memory', ax=ax, style='*-')
plt.legend(["GPU5","CPU"],loc='upper left')
plt.xlabel('Memory Footprint (Mbyte)')
plt.ylabel('Performance (Gflops/s)')
#plt.gca().set_ylim(bottom=0)
plt.xscale('log',basex=2)
plt.yscale('log',basey=2)
ax = plt.gca().xaxis
ax.set_major_formatter(FormatStrFormatter('%.2f'))
ax = plt.gca().yaxis
ax.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(rotation=45)
plt.savefig('Mflops_GPU5_matmul.png', bbox_inches='tight')
plt.close()

plt.figure()
ax = df7.plot('memory',logx=False, style='.-')
df6.plot('memory', ax=ax, style='*-')
plt.legend(["cublasDgemm","CPU"],loc='upper left')
plt.xlabel('Memory Footprint (Mbyte)')
plt.ylabel('Performance (Gflops/s)')
#plt.gca().set_ylim(bottom=0)
plt.xscale('log',basex=2)
plt.yscale('log',basey=2)
ax = plt.gca().xaxis
ax.set_major_formatter(FormatStrFormatter('%.2f'))
ax = plt.gca().yaxis
ax.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xticks(rotation=45)
plt.savefig('Mflops_cublas_matmul.png', bbox_inches='tight')
plt.close()

plt.figure()
df1['GPU1/CPU'] = df1['Mflop/s']/df6['Mflop/s']
df2['GPU2/CPU'] = df2['Mflop/s']/df6['Mflop/s']
df3['GPU3/CPU'] = df3['Mflop/s']/df6['Mflop/s']
df4['GPU4/CPU'] = df4['Mflop/s']/df6['Mflop/s']
df5['GPU5/CPU'] = df5['Mflop/s']/df6['Mflop/s']
df7['cublas/CPU'] = df7['Mflop/s']/df6['Mflop/s']
ax = df1["GPU1/CPU"].plot(legend=True, style ='*-')
ax = df2["GPU2/CPU"].plot(legend=True, style ='*-')
ax = df3["GPU3/CPU"].plot(legend=True, style ='*-')
ax = df4["GPU4/CPU"].plot(legend=True, style ='*-')
ax = df5["GPU5/CPU"].plot(legend=True, style ='*-')
ax = df7["cublas/CPU"].plot(legend=True, style ='*-')
plt.legend(loc='lower right')
plt.xlabel('Memory footprint (Mbytes)')
plt.ylabel('Performance boost')
plt.xscale('log',basex=2)
plt.yscale('log',basey=2)
plt.gca().set_ylim(bottom=0)
ax = plt.gca().xaxis
ax.set_major_formatter(FormatStrFormatter('%.2f'))
ax = plt.gca().yaxis
ax.set_major_formatter(FormatStrFormatter('%.4f'))
plt.savefig('Mflops_speedup.png', bbox_inches='tight')
plt.close()
