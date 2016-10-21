# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

df = pd.read_csv('Data.csv')
df.index = df['Ticker']
df = df.T
df = df.drop('Ticker')

clm_drop = ['Ticker', 'Currency', 'Name', 'Ratings', 'Filing Date', 'End of Next Annual Period','End of Period', 'Report Date']
_trData = df.drop(clm_drop, 1)

trData = _trData[_trData['Ratings #'].notnull()]
testData = _trData[_trData['Ratings #'].isnull()]

trData.to_csv('TrainingData.csv', index = None)
testData.to_csv('testData.csv', index = None)
