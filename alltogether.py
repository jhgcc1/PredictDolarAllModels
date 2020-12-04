
from scrap import getData
from model import mlModel2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from preprocessClass import PreProcess
import xgboost as xgb

years=1

consider=["divida/pib eua","Relacao divida/pib Brasil","dxy","brent",'gold','bovespa','vix','usd_cny','usd_cop','usd_mxn','usd_brl',"Fed interest rate decisions","Selic"]
considerModels=["xg","en","gb","svr"]
#considerModels=["svr"]
normType=["MinMaxScaler","StandardScaler","MaxAbsScaler","RobustScaler","QuantileTransformerNormal","QuantileTransformerUniform","PowerTransformer"]
df=getData(years,consider)
print(df)
gm_cv,betspar,r2,mse,residual,estimator,norm,X_train, y_train=mlModel2(df,"r2",considerModels,normType)

sevendaysdf=df.loc[df.index >= len(df.index)-40, df.columns!="usd_brl"]
sevendaysdfY=df.loc[df.index >= len(df.index)-40, df.columns=="usd_brl"]
sevendaysPredict=gm_cv.predict(sevendaysdf)
print(sevendaysdf)
print(sevendaysPredict)

norm__classifier_type=betspar["norm__classifier_type"]
clf__estimator=betspar["clf__estimator"]



sevendaysdfNorm = norm.transform(sevendaysdf)

x = list(range(len(sevendaysdfNorm)))
plt.figure(figsize=(12,8))
if type(clf__estimator)==type(GradientBoostingRegressor()):
    clf__estimator__learning_rate=betspar['clf__estimator__learning_rate']
    mid=sevendaysPredict
    lower = Pipeline([('norm', PreProcess(norm__classifier_type)), ('clf', GradientBoostingRegressor(alpha=0.1,learning_rate=clf__estimator__learning_rate,loss="quantile"))])
    lower.fit(X_train, y_train)
    higher = Pipeline([('norm', PreProcess(norm__classifier_type)), ('clf', GradientBoostingRegressor(alpha=0.9,learning_rate=clf__estimator__learning_rate,loss="quantile"))])
    higher.fit(X_train, y_train)
    higherData=higher.predict(sevendaysdf)
    lowerData=lower.predict(sevendaysdf)

    plt.plot(x,higherData, color='grey', alpha=0.2, zorder=1)
    plt.plot(x, lowerData, color='grey', alpha=0.2, zorder=1)
    plt.plot(x,mid, color='red', zorder=5)
    plt.scatter(x,sevendaysdfY, marker='o', color='orange', zorder=4)
elif type(clf__estimator)==type(xgb.XGBRegressor()):
    print("innnn")
    plt.scatter(x,sevendaysdfY, marker='o', color='orange', zorder=4)

    # "Bagging model" prediction
    plt.plot(x,sevendaysPredict, color='red', zorder=5)

    plt.savefig('testplot.png')
else:
    for m in estimator.estimators_:
        print(m.coef_)
        print(m.intercept_)
        print(m.predict(sevendaysdfNorm))
        print(m)
        print(m.get_params())
        plt.plot(x, m.predict(sevendaysdfNorm), color='grey', alpha=0.2, zorder=1)

    plt.scatter(x,sevendaysdfY, marker='o', color='orange', zorder=4)

    # "Bagging model" prediction
    plt.plot(x,sevendaysPredict, color='red', zorder=5)

    plt.savefig('testplot.png')