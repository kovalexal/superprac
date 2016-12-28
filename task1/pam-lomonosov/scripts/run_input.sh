#!/bin/bash

# $1 -- queue
# $2 -- processes
# $3 -- time
# $4 -- input_start
# $5 -- input_step
# $6 -- input_end

for i in $(seq $4 $5 $6)
do 
    sbatch -p $1 -n$2 -t$3 -o ../cout/20_$i_$2.txt ompi ../pam-lomonosov ../data/20_$i_distances.csv 20 1000 ../results_new/20_$i_$2.json
done
