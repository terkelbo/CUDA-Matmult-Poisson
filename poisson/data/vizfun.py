import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import numpy as np

font = {'size'   : 14}

mpl.rc('font', **font)

df1 = pd.read_csv("statfun_poisson_CPU.dat",delim_whitespace=True,header=None,names=["MaxIter","Memory","MFlops","WallTime","Bandwidth","Size"])
df1 = df1.loc[(df1['Size'] >= 10) & (df1['Size'] <= 10000)]
df2 = pd.read_csv("statfun_poisson_gpu1.dat",delim_whitespace=True,header=None,names=["MaxIter","Memory","MFlops","WallTime","Bandwidth","Size"])
df2 = df2.loc[(df2['Size'] >= 10) & (df2['Size'] <= 1250)]
df3 = pd.read_csv("statfun_poisson_gpu2.dat",delim_whitespace=True,header=None,names=["MaxIter","Memory","MFlops","WallTime","Bandwidth","Size"])
df3 = df3.loc[(df3['Size'] >= 10) & (df3['Size'] <= 10000)]
df4 = pd.read_csv("statfun_poisson_gpu3.dat",delim_whitespace=True,header=None,names=["MaxIter","Memory","MFlops","WallTime","Bandwidth","Size"])
df4 = df4.loc[(df4['Size'] >= 10) & (df4['Size'] <= 10000)]

plt.figure()
df1["MFlops"]=df1["MFlops"]/1000
df2["MFlops"]=df2["MFlops"]/1000
df3["MFlops"]=df3["MFlops"]/1000
df4["MFlops"]=df4["MFlops"]/1000
df1.set_index("Memory", inplace=True)
df2.set_index("Memory", inplace=True)
df3.set_index("Memory", inplace=True)
df4.set_index("Memory", inplace=True)
ax = df1["MFlops"].plot(legend=True, style ='*-')
df2["MFlops"].plot(legend=True, style ='*-',ax=ax)
df3["MFlops"].plot(legend=True, style ='*-',ax=ax)
df4["MFlops"].plot(legend=True, style ='*-',ax=ax)
plt.legend(["CPU","GPU1","GPU2","GPU3"],loc='upper left')
plt.xlabel('Memory Footprint (Kbyte)')
plt.ylabel('Performance (GFlops)')
plt.gca().set_ylim(bottom=0)
plt.xscale('log',basex=4)
#plt.yscale('log',basey=2)
ax = plt.gca().xaxis
#ax.set_major_formatter(ScalarFormatter())
ax.set_major_formatter(FormatStrFormatter('%.f'))
ax = plt.gca().yaxis
ax.set_major_formatter(ScalarFormatter())
plt.savefig('MFlops.png', bbox_inches='tight')
plt.close()


plt.figure()
ax = df1["WallTime"].plot(legend=True, style ='*-')
df2["WallTime"].plot(legend=True, style ='*-',ax=ax)
df3["WallTime"].plot(legend=True, style ='*-',ax=ax)
df4["WallTime"].plot(legend=True, style ='*-',ax=ax)
plt.legend(["CPU","GPU1","GPU2","GPU3"],loc='upper left')
plt.xlabel('Memory Footprint (Kbyte)')
plt.ylabel('Elapsed Wall Clock Time (s)')
plt.gca().set_ylim(bottom=0)
plt.xscale('log',basex=4)
#plt.yscale('log',basey=2)
ax = plt.gca().xaxis
#ax.set_major_formatter(ScalarFormatter())
ax.set_major_formatter(FormatStrFormatter('%.f'))
#ax = plt.gca().yaxis
#ax.set_major_formatter(ScalarFormatter())
plt.savefig('WallTime.png', bbox_inches='tight')
plt.close()
                                                             

plt.figure()
df1["Bandwidth"]=df1["Bandwidth"]/1000
df2["Bandwidth"]=df2["Bandwidth"]/1000
df3["Bandwidth"]=df3["Bandwidth"]/1000
df4["Bandwidth"]=df4["Bandwidth"]/1000
ax = df1["Bandwidth"].plot(legend=True, style ='*-')
df2["Bandwidth"].plot(legend=True, style ='*-',ax=ax)
df3["Bandwidth"].plot(legend=True, style ='*-',ax=ax)
df4["Bandwidth"].plot(legend=True, style ='*-',ax=ax)
plt.legend(["CPU","GPU1","GPU2","GPU3"],loc='upper left')
plt.xlabel('Memory footprint (Kbytes)')
plt.ylabel('Bandwidth (GB/s)')
#plt.gca().set_ylim(bottom=0)
plt.xscale('log',basex=4)
plt.yscale('log',basey=2)
ax = plt.gca().xaxis
ax.set_major_formatter(FormatStrFormatter('%.f'))
#ax.set_major_formatter(ScalarFormatter())
ax = plt.gca().yaxis
ax.set_major_formatter(ScalarFormatter())
plt.savefig('Bandwidth.png', bbox_inches='tight')
plt.close()
                                                                                                                                    

plt.figure()
df1 = df1.sort_values(['Size'],ascending=True)
df2 = df2.sort_values(['Size'],ascending=True)
df3 = df3.sort_values(['Size'],ascending=True)
df4 = df4.sort_values(['Size'],ascending=True)
df2['CPU/GPU1'] = df1.loc[df1['Size'] <= 1250]['WallTime']/df2['WallTime']
df3['CPU/GPU2'] = df1['WallTime']/df3['WallTime']
df4['CPU/GPU3'] = df1['WallTime']/df4['WallTime']
ax = df2["CPU/GPU1"].plot(legend=True, style ='*-')
ax = df3["CPU/GPU2"].plot(legend=True, style ='*-')
ax = df4["CPU/GPU3"].plot(legend=True, style ='*-')
plt.legend(loc='upper left')
plt.xlabel('Memory footprint (kbytes)')
plt.ylabel('Speedup')
plt.xscale('log',basex=4)
plt.yscale('log',basey=2)
ax = plt.gca().xaxis
#ax.set_major_formatter(ScalarFormatter())
ax.set_major_formatter(FormatStrFormatter('%.f'))
ax = plt.gca().yaxis
ax.set_major_formatter(FormatStrFormatter('%.2f'))
plt.savefig('Speedup.png', bbox_inches='tight')
plt.close()
	