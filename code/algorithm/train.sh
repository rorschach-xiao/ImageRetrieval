#! /bin/bash

cd ~/ImageRetrieval/code/algorithm
python train.py -queryindex ../../data/features/featureCNN_Q_big.h5 -index ../../data/features/featureCNN_40.h5 -negativeindex ../../data/features/featureCNN_neg.h5 -model ../../models/model40/ -MaxRes 40 -trainmode 1 
