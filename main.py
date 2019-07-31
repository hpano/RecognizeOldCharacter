"""
This is a sample script for PRMU Algorithm Contest 2019.
You can modify the codes anywhere.
Make sure the output format.
"""

import numpy as np
import pandas as pd
from srcs.utils import AlconDataset, AlconTargets, unicode_to_kana_list, evaluation
from srcs.myalgorithm import MyAlgorithm

"""
Main Script
"""
# Set dataset path manually
datapath = './dataset/'

# Load dataset
targets = AlconTargets(datapath, train_ratio=0.99)
traindata = AlconDataset(datapath, targets.train, isTrainVal=True)
valdata = AlconDataset(datapath, targets.val, isTrainVal=True)
testdata = AlconDataset(datapath, targets.test, isTrainVal=False)

# Set params
max_epochs = 200
num_train = 12000
batch_size = 900
img_size = 32

# Train model
print('Building model ...')
myalg = MyAlgorithm()  # Your algorithm class
myalg.build_model(traindata, valdata, max_epochs, num_train, batch_size, img_size)  # Train my model
print('done')

"""
# Prediction
N = len(valdata)
sheet = valdata.getSheet()  # Get initial sheet
for i in range(N):
    # Prediction
    img, y_val = valdata[i]  # Get data
    y_pred = myalg.predict(img)  # Prediction
    print('Prediction {}; GT {}; {}/{}'.format(
        unicode_to_kana_list(y_pred), unicode_to_kana_list(y_val), i, N))

    # Fill the sheet with y_pred
    sheet.iloc[i, 1:4] = y_pred

# Evaluation
acc = evaluation(sheet, valdata.targets)
print('Accuracy = {:f} (%)'.format(acc))

# save predicted results in CSV
# Zip and submit it.
sheet.to_csv('test_prediction.csv', index=False)
"""
