#!/bin/bash

for i in $(seq $4 $5 $6)
do 
    sbatch -p $2 -n$i -t$3 -o ../cout/$1_$i.txt ompi ../pam-lomonosov ../data/$1_distances.csv 20 1000 ../results_small/$1_$i.json
done
