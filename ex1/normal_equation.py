import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y   #X.T@X等价于X.T.dot(X)
    return theta