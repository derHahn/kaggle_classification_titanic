import sys
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, KBinsDiscretizer, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

def extract_labels(train_path, test_path, path_X_train, path_y_train, path_X_test, path_y_test):
    """
    Extract labels y from train and test

    train_path: str
        file path to train.csv
    test_path: str
        file path to test.csv
    """
    df_train = pd.read_csv(train_path, index_col=0)
    df_test = pd.read_csv(test_path, index_col=0)

    # training set
    X_train = df_train.drop('Survived', axis=1)
    y_train = df_train['Survived']
    X_train.to_csv(path_X_train)
    y_train.to_csv(path_y_train)

    # test set
    X_test = df_test.drop('Survived', axis=1)
    y_test = df_test['Survived']
    X_test.to_csv(path_X_test)
    y_test.to_csv(path_y_test)

def cabin_code(frame):
    """get one letter code for cabin, add 'U' for unknown"""
    column = frame.iloc[:,0]
    column.fillna('U', inplace=True)
    return column.str[0].to_frame()

def family(frame):
    """add a column with sum of family members"""
    frame['family'] = frame.sum(axis=1)
    frame['alone'] = (frame['family'] > 1).astype(int)
    return frame

def title_len(frame):
    """add column with length of name"""
    column = frame.iloc[:,0]
    return column.str.len().to_frame()

def titles(frame):
    """extract titles from names"""
    output = frame.copy()
    col_name = output.columns[0]
    for i in output.index:
        name = str(output.loc[i, col_name])
        name = name.replace(',', '')
        name = name.replace('(', '')
        name = name.replace(')', '')
        name = name.replace('"', '')
        name = name.split(' ')
        if 'Mr.' in name or 'Mr ' in name:
            output.loc[i] = 'Mr'
        elif 'Miss' in name:
            output.loc[i] = 'Miss'
        elif 'Mrs.' in name or 'Mrs ' in name:
            output.loc[i] = 'Mrs'
        elif 'Master' in name:
            output.loc[i] = 'Master'
        elif 'Dr.' in name:
            output.loc[i] = 'Dr'
        elif 'Jr' in name or 'Jr.' in name:
            output.loc[i] = 'Jr'
        else:
            output.loc[i] = 'other'
    return output

def add_bias(frame):
    """add bias for box-cox transformation, all > 0"""
    frame.fillna(0, inplace=True)
    return frame + 0.001

def make_features(path_X_train, path_X_test, X_train_features_path, X_test_features_path):
    """
    path_X_train: str
        file path to X_train
    seed: int
        random seed
    """
    age_pipe = Pipeline([
        ('age_imp', SimpleImputer(strategy='mean')),
        ('age_bin', KBinsDiscretizer(encode='ordinal', strategy='quantile', n_bins=3)),
        ('age_ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    title_pipe = Pipeline([
        ('title_get', FunctionTransformer(titles)),
        ('titles_ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    family_pipe = Pipeline([
        ('fam_get', FunctionTransformer(family)),
        ('fam_ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    cabin_pipe = Pipeline([
        ('cab_letter', FunctionTransformer(cabin_code)),
        ('cab_ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    embarked_pipe = Pipeline([
        ('emb_imp', SimpleImputer(strategy='most_frequent')),
        ('emb_ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    fare_pipe = Pipeline([
        ('fare_add', FunctionTransformer(add_bias)),
        ('fare_trans', PowerTransformer(method='box-cox')),
        ('fare_bin', KBinsDiscretizer(encode='ordinal')),
        ('fare_ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    ct = ColumnTransformer([
        ('cabin', cabin_pipe, ['Cabin']),
        ('family', family_pipe, ['SibSp', 'Parch']),
        ('name_len', FunctionTransformer(title_len), ['Name']),
        ('title', title_pipe, ['Name']),
        ('fare', fare_pipe, ['Fare']),
        ('age', age_pipe, ['Age']),
        ('class', OneHotEncoder(), ['Pclass']),
        ('sex', OneHotEncoder(), ['Sex']),
        ('embark', embarked_pipe, ['Embarked'])
    ], remainder='drop')

    X_train = pd.read_csv(path_X_train, index_col=0)
    X_test = pd.read_csv(path_X_test, index_col=0)
    X_train_featurized = ct.fit_transform(X_train)
    X_test_featurized = ct.transform(X_test)
    np.savetxt(X_train_features_path, X_train_featurized.todense(), delimiter=',')
    np.savetxt(X_test_features_path, X_test_featurized.todense(), delimiter=',')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        X_train_path = sys.argv[3]
        y_train_path = sys.argv[4]
        X_test_path = sys.argv[5]
        y_test_path = sys.argv[6]
        X_train_features_path = sys.argv[7]
        X_test_features_path = sys.argv[8]
    else:
        train_path = 'data/processed/train.csv'
        test_path = 'data/processed/test.csv'
        X_train_path = 'data/processed/X_train.csv'
        y_train_path = 'data/processed/y_train.csv'
        X_test_path = 'data/processed/X_test.csv'
        y_test_path = 'data/processed/y_test.csv'
        X_train_features_path = 'data/featurized/X_train.csv'
        X_test_features_path = 'data/featurized/X_test.csv'

    extract_labels(train_path, test_path, X_train_path, y_train_path, X_test_path, y_test_path)
    make_features(X_train_path, X_test_path, X_train_features_path, X_test_features_path)