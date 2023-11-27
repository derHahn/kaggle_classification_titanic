import sys
import dvc.api
import logging
import joblib
import pandas as pd 
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

def train_model(path_X_train, path_y_train, path_model_obj):
    X_train = pd.read_csv(path_X_train, index_col=0, header=None)
    y_train = pd.read_csv(path_y_train, index_col=0)

    if model_type == 'logreg':
        model = LogisticRegression()
    elif model_type == 'forest':
        model = RandomForestClassifier()
    elif model_type == 'extra':
        model = ExtraTreesClassifier()
    elif model_type == 'gradient':
        model = GradientBoostingClassifier()
    else:
        logging.error('Model type not found')

    model.fit(X_train, y_train.to_numpy().reshape(-1, 1))
    joblib.dump(model, path_model_obj)    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        params = dvc.api.params_show()
        model_type = params['train']['model_type']
        path_X_train = sys.argv[1] 
        path_y_train = sys.argv[2]
        path_model_obj = sys.argv[3]
    else:
        model_type = 'logreg'
        path_X_train = 'data/featurized/X_train.csv'
        path_y_train = 'data/featurized/y_train.csv'
        path_model_obj = 'models/model.joblib'

    train_model(path_X_train, path_y_train, path_model_obj)