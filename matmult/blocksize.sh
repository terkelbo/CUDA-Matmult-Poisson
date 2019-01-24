#!/bin/sh

blocksize=32
make clean && make 
for size in 4000 8000
do 
	output=$(MATMULT_COMPARE=0 ./matmult_f.nvcc gpu5 $size $size $size)
	echo "$blocksize $output $size" >> data/sharedmem_blocksize.dat	
done
