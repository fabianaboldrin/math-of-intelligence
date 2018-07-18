import numpy as np
import pandas as pd
from patsy import dmatrices
import warnings

def sigmoid(x):
    '''SIGMOID FUNCTION FOR X'''

    return 1/(1+np.exp(-x))