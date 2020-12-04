from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, MaxAbsScaler, RobustScaler, PowerTransformer
class PreProcess(BaseEstimator):

    def __init__(self, classifier_type: str = 'MinMaxScaler'):

        self.classifier_type = classifier_type

    def fit(self, X,y=None):
        if self.classifier_type == 'StandardScaler':
            self.classifier_ = StandardScaler()
        elif self.classifier_type == 'MinMaxScaler':
            self.classifier_ = MinMaxScaler()
        elif self.classifier_type == 'MaxAbsScaler':
            self.classifier_ = MaxAbsScaler()
        elif self.classifier_type == 'RobustScaler':
            self.classifier_ = RobustScaler()
        elif self.classifier_type == 'QuantileTransformerUniform':
            self.classifier_ = QuantileTransformer(output_distribution="uniform")
        elif self.classifier_type == 'QuantileTransformerNormal':
            self.classifier_ = QuantileTransformer(output_distribution="normal")
        elif self.classifier_type == 'PowerTransformer':
            self.classifier_ = PowerTransformer(method="yeo-johnson")
        else:
            raise ValueError('Unkown classifier type.')
        self.classifier_.fit(X)
        return self
    def transform(self,X,y=None):
        return self.classifier_.transform(X)