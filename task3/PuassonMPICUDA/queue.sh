#!/bin/sh
squeue -i 2 -h -l --format "%.7i %.9P %.4j %.15u %.8T %.12M %.12l %.8p %.20S %C %R" --states=RUNNING,PENDING -u kovalexal_1854

