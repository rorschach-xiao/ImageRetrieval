#! /bin/bash

cd ~/ImageRetrieval/code/algorithm
python query.py -queryindex ../../data/features/featureCNN_Q_big.h5 -index ../../data/features/featureCNN_big.h5 -model ../../models/model60/ -MaxRes 60 -result result1.csv 
