#!/bin/bash

#BSUB -q hpcintro
#BSUB -J CPU
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -M 3GB
#BSUB -W 01:00
#BSUB -N
#BSUB -oo CPU.out
#BSUB -eo CPU.err

module load cuda/10.0
module load gcc/7.3.0


N="10 20 30 50 100 150 200 250 500 1000 1250 1500 2000 2500 2830"
IMPLEMENTATIONS="poisson_cpu"

rm -f data/statfun_$IMPLEMENTATIONS.dat
echo "Making data file containing stats - see data/statfun_$IMPLEMENTATIONS.dat"

do 
for rows in $N
	do
	echo "OMP_NUM_THREADS=12 numactl --cpunodebind=0 ./$IMPLEMENTATIONS $rows"
	output=$(OMP_NUM_THREADS=12 numactl --cpunodebind=0 ./$IMPLEMENTATIONS $rows)
	echo "$t $output $rows" >> data/statfun_$IMPLEMENTATIONS.dat
done



echo "File is done"

exit 0

