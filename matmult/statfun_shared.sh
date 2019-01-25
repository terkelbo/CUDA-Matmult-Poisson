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


func="gpu5"
N="16 64 128 512 1024 1536 2048 4096 6144 8192 10240 14336 20480 24576"

rm -f ./data/statfun_$func.dat
for n in $N
do
	output=$(MATMULT_COMPARE=0 ./matmult_f.nvcc $func $n $n $n)
	echo "$output $n" >> ./data/statfun_$func.dat
done

echo "File is done"

exit 0

