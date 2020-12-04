from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from preprocessClass import PreProcess
from regressionModelClass import ClfSwitcher
import xgboost as xgb

l1_space = np.linspace(0, 1, 5)
alpha=np.logspace(-1,1,5)
linspace01=np.linspace(0,1,5)
print(l1_space)
print(alpha)
print(linspace01)