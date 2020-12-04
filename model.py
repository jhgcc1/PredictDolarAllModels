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

def mlModel2(df,scoreMethod,considerModels,normType):
    dictModels={}
    l1_space = np.linspace(0, 1, 5)
    alpha=np.logspace(-1,1,5)
    linspace01=np.linspace(0,1,3)
    dictModels["gb"]={'clf__estimator': [GradientBoostingRegressor(loss="quantile",alpha=0.5)],
            'clf__estimator__learning_rate': [0.05,0.1, 0.15,0.2],
            "norm__classifier_type":normType}

    dictModels["en"]={'clf__estimator': [BaggingRegressor(ElasticNet(tol=1), n_estimators=20)],
            "clf__estimator__base_estimator__alpha":alpha,
            'clf__estimator__base_estimator__l1_ratio': l1_space,
            "norm__classifier_type":normType
        }
    dictModels["svr"]={
            'clf__estimator': [BaggingRegressor(SVR(tol=1,kernel="linear"), n_estimators=20)],
            'clf__estimator__base_estimator__C': [0.1, 2],
            #clf__estimator__base_estimator__epsilon': [ 0.01, 2],
            'clf__estimator__base_estimator__gamma': [ 0.1, 2],
            "norm__classifier_type":normType
        }
    dictModels["xg"]={
            'clf__estimator': [xgb.XGBRegressor(objective ='reg:squarederror', n_estimators = 20)],
            "norm__classifier_type":normType,
            'clf__estimator__learning_rate': [0.15,0.2,0.3],
            'clf__estimator__colsample_bytree':linspace01,
            'clf__estimator__colsample_bylevel':linspace01,
            #'clf__estimator__reg_alpha':l1_space,
            #'clf__estimator__reg_lambda':l1_space
            
        }
        
    param_grid=[dictModels[model] for model in dictModels if model in considerModels]
    print(param_grid)

    pipe = Pipeline([('norm', PreProcess()), ('clf', ClfSwitcher())])


    gm_cv = GridSearchCV(pipe, param_grid, cv=5,refit=True,scoring=scoreMethod)
    X = df.loc[:,df.columns!="usd_brl"]
    Y = df['usd_brl']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state=42)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    gm_cv.fit(X_train, y_train)

    y_pred = gm_cv.predict(X_test)
    residual = (y_test - y_pred)
    r2 = gm_cv.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    betspar = gm_cv.best_params_
    est = gm_cv.best_estimator_.named_steps['clf'].estimator
    norm = gm_cv.best_estimator_.named_steps['norm']

    print("Tuned ElasticNet best estimators: {}".format(betspar))
    print("Tuned ElasticNet R squared: {}".format(r2))
    print("Tuned ElasticNet MSE: {}".format(mse))
    return gm_cv,betspar,r2,mse,residual,est,norm,X_train, y_train
