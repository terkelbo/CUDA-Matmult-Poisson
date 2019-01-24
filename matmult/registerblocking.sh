#!/bin/sh

make clean && make

output=$(./matmult_f.nvcc gpu4 10000 10000 10000)
echo "
