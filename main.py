"""
This is a script for PRMU Algorithm Contest 2019.
"""

import numpy as np
import pandas as pd
from srcs.utils import AlconDataset, AlconTargets, unicode_to_kana_list, evaluation
from srcs.myalgorithm import MyAlgorithm


# Set dataset path manually
datapath = './dataset/'

# Load dataset
targets = AlconTargets(datapath, train_ratio=0.99)
traindata = AlconDataset(datapath, targets.train, isTrainVal=True)
valdata = AlconDataset(datapath, targets.val, isTrainVal=True)
testdata = AlconDataset(datapath, targets.test, isTrainVal=False)

# Set params
max_epochs = 3
num_train = 12000
batch_size = 15
img_size = 32

params_name = "params_100_12000_900_1564640851.pkl"
isTrain = False
myalg = MyAlgorithm(traindata, valdata, testdata, max_epochs, num_train, batch_size, img_size, params_name)

# Train model
if isTrain:
    print("##### Building model #####")
    myalg.build_model()
    print("done")

# Prediction
print("##### Predicting testdata #####")
myalg.predict()
print("done, save prediction as 'test_prediction.csv'")
