#!/bin/bash

#BSUB -q hpcintrogpu
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -J GPU2
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -M 3GB
#BSUB -W 01:00
#BSUB -N
#BSUB -oo GPU2.out
#BSUB -eo GPU2.err

module load cuda/10.0
module load gcc/7.3.0



N="10 20 30 50 100 150 200 250 500 1000 1250 1500 2000 2500 2830"
IMPLEMENTATIONS="poisson_gpu2"

rm -f data/statfun_$IMPLEMENTATIONS.dat
echo "Making data file containing stats - see data/statfun_$IMPLEMENTATIONS.dat"

do 
for rows in $N
	do
	echo "./$IMPLEMENTATIONS $rows"
	output=$(./$IMPLEMENTATIONS $rows)
	echo "$t $output $rows" >> data/statfun_$IMPLEMENTATIONS.dat
done



echo "File is done"

exit 0

