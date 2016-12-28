#!/bin/bash

for i in $(seq 250 250 5000)
do 
    ./generate.py 20 $i
done