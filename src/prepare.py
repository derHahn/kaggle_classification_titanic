import dvc.api
import sys
import logging

import pandas as pd 

from sklearn.model_selection import train_test_split

def split_data(raw_data_path, seed, train_size):
    """
    Split the data in train and test and save the processed data

    Parameters
    ----------
    raw_data_path : str
        path to the raw data
    seed : int
        random seed
    train_size : float
        ratio of train and test
    """
    raw_data = pd.read_csv(raw_data_path, index_col=0)

    train, test = train_test_split(raw_data, train_size=train_size, random_state=seed)
    train.to_csv('data/processed/train.csv')
    test.to_csv('data/processed/test.csv')
    logging.info("Data processed (splitted)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        raw_data_path = sys.argv[1]
        params = dvc.api.params_show()
        seed = params['prepare']['seed']
        train_size = params['prepare']['split']
    else: 
        logging.error("Couldn't use parameters from outside, falling to test mode")
        raw_data_path = "data/raw/train.csv"
        seed = 12
        train_size = 0.8
    
    split_data(raw_data_path, seed, train_size)