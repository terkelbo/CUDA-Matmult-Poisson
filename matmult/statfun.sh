#!/bin/bash

#BSUB -q hpcintrogpu
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -J statfun
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 01:00
#BSUB -N
#BSUB -oo statfun.out
#BSUB -eo statfun.err


func="gpu1"
N="10 50 100 500 1000 1500 2000 4000 6000 8000"

rm -f ./data/statfun_$func.dat
for n in $N
do
	output=$(MATMULT_COMPARE=0 ./matmult_f.nvcc $func $n $n $n)
	echo "$output $n" >> ./data/statfun_$func.dat
done

echo "File is done"

exit 0

