import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


datafile = 'data/ex1data1.txt'
data = pd.read_csv(datafile, header=None, names=['Population', 'Profit'])

data.head()

data.describe()

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show()