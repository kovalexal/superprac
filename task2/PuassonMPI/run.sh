#!/bin/sh
sbatch -n$1 --ntasks-per-node=2 --time=$2 -p $4 impi ./main.out $3 $3 1 100000
