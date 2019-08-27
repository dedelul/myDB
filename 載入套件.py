import pyximport; pyximport.install()
import gc
import time
import numpy as np
import pandas as pd
# from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDRegressor
import lightgbm as lgb
from PyQt5 import QtCore, QtGui
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
%matplotlib inline
