import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import numpy as np

font = {'size'   : 14}

mpl.rc('font', **font)

for files, names in zip(['statfun_poisson_CPU.dat','statfun_poisson_gpu1.dat','statfun_poisson_gpu2.dat','statfun_poisson_gpu3.dat'],
						['statfun_poisson_CPU','statfun_poisson_gpu1','statfun_poisson_gpu2','statfun_poisson_gpu3']):
	df1 = pd.read_csv(files,delim_whitespace=True,header=None,names=["MaxIter","Memory","MFlops","WallTime","Bandwidth","Size"])

	plt.figure()
	df1.set_index("Memory", inplace=True)
	ax = df1["MFlops"].plot(legend=True, style ='*-')
	plt.legend(loc='upper left')
	plt.xlabel('Memory Footprint (Kbyte)')
	plt.ylabel('Performance (MFlops)')
	plt.gca().set_ylim(bottom=0)
	plt.xscale('log',basex=4)
	#plt.yscale('log',basey=2)
	ax = plt.gca().xaxis 
	if "gpu3" not in files:
		ax.set_major_formatter(ScalarFormatter()) 
	#ax = plt.gca().yaxis 
	#ax.set_major_formatter(ScalarFormatter()) 
	plt.savefig('MFlops_%s.png' % names, bbox_inches='tight')
	plt.close()

	plt.figure()
	ax = df1["WallTime"].plot(legend=True, style ='*-')
	plt.legend(loc='upper left')
	plt.xlabel('Memory Footprint (Kbyte)')
	plt.ylabel('Elapsed Wall Clock Time (s)')
	plt.gca().set_ylim(bottom=0)
	plt.xscale('log',basex=4)
	#plt.yscale('log',basey=2)
	ax = plt.gca().xaxis
	if "gpu3" not in files:
		ax.set_major_formatter(ScalarFormatter())
	#ax = plt.gca().yaxis
	#ax.set_major_formatter(ScalarFormatter())
	plt.savefig('WallTime_%s.png' % names, bbox_inches='tight')
	plt.close()

	df1 = pd.read_csv(files,delim_whitespace=True,header=None,names=["MaxIter","Memory","MFlops","WallTime","Bandwidth","Size"])
	df1 = df1.loc[df1['Size'] >= 100]
	plt.figure()
	df1.set_index("Memory", inplace=True)
	ax = df1["Bandwidth"].plot(legend=True, style ='*-')
	plt.legend(loc='upper left')
	plt.xlabel('Memory footprint (Kbytes)')
	plt.ylabel('Bandwidth (Mflops/s)')
	#plt.gca().set_ylim(bottom=0)
	plt.xscale('log',basex=2)
	plt.yscale('log',basey=2)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	ax = plt.gca().xaxis
	if "gpu3" not in files: 
		ax.set_major_formatter(ScalarFormatter()) 
	ax = plt.gca().yaxis 
	#ax.set_major_formatter(ScalarFormatter()) 
	plt.savefig('Bandwidth_%s.png' % names, bbox_inches='tight')
	plt.close()

	
df1 = pd.read_csv("statfun_poisson_CPU.dat",delim_whitespace=True,header=None,names=["MaxIter","Memory","MFlops","WallTime","Bandwidth","Size"])
df1 = df1.loc[(df1['Size'] >= 100) & (df1['Size'] < 2900)]
df2 = pd.read_csv("statfun_poisson_gpu1.dat",delim_whitespace=True,header=None,names=["MaxIter","Memory","MFlops","WallTime","Bandwidth","Size"])
df2 = df2.loc[(df2['Size'] >= 100) & (df2['Size'] < 2900)]
df3 = pd.read_csv("statfun_poisson_gpu2.dat",delim_whitespace=True,header=None,names=["MaxIter","Memory","MFlops","WallTime","Bandwidth","Size"])
df3 = df3.loc[(df3['Size'] >= 100) & (df3['Size'] < 2900)]
df4 = pd.read_csv("statfun_poisson_gpu3.dat",delim_whitespace=True,header=None,names=["MaxIter","Memory","MFlops","WallTime","Bandwidth","Size"])
df4 = df4.loc[(df4['Size'] >= 100) & (df4['Size'] < 2900)]

plt.figure()
df1 = df1.sort_values(['Size'],ascending=True)
df2 = df2.sort_values(['Size'],ascending=True)
df3 = df3.sort_values(['Size'],ascending=True)
df4 = df4.sort_values(['Size'],ascending=True)

#df2['CPU/GPU1'] = df1['WallTime']/df2['WallTime']
df3['CPU/GPU2'] = df1['WallTime']/df3['WallTime']
df4['CPU/GPU3'] = df1['WallTime']/df4['WallTime']

df2.set_index("Memory", inplace=True)
df3.set_index("Memory", inplace=True)
df4.set_index("Memory", inplace=True)

#ax = df2["Speedup"].plot(legend=True, style ='*-')
ax = df3["CPU/GPU2"].plot(legend=True, style ='*-')
ax = df4["CPU/GPU3"].plot(legend=True, style ='*-')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Memory footprint (kbytes)')
plt.ylabel('Speedup')
plt.xscale('log',basex=2)
plt.yscale('log',basey=2)
ax = plt.gca().xaxis
#ax.set_major_formatter(ScalarFormatter())
ax = plt.gca().yaxis
ax.set_major_formatter(FormatStrFormatter('%.2f'))
plt.savefig('Speedup.png', bbox_inches='tight')
plt.close()
	
