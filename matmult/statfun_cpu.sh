#!/bin/bash

#BSUB -q hpcintro
#BSUB -J statfun
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=3GB]"
#BSUB -W 02:00
#BSUB -N
#BSUB -oo statfun.out
#BSUB -eo statfun.err


func="lib"
N="10 50 100 500 1000 1500 2000 4000 6000 8000 10000 15000 20000 25000"

rm -f ./data/statfun_$func.dat
for n in $N
do
	output=$(OMP_NUM_THREADS=12 MATMULT_COMPARE=0 numactl --cpunodebind=0 ./matmult_f.nvcc $func $n $n $n)
	echo "$output $n" >> ./data/statfun_$func.dat
done

echo "File is done"

exit 0

