import pandas as pd
import numpy as np
from logging import getLogger

logger = getLogger(__name__)

TRAIN_DATA = './input/train.csv'
TEST_DATA = './input/test.csv'

def read_csv(path):
    logger.debug('start read csv')
    df = pd.read_csv(path)
    logger.debug('end read csv')
    return df

def load_train_csv():
    logger.debug('start read csv')
    df = pd.read_csv(TRAIN_DATA)
    logger.debug('end read csv')
    return df

def load_test_csv():
    logger.debug('start read csv')
    df = pd.read_csv(TEST_DATA)
    logger.debug('end read csv')
    return df

if __name__ == '__main__':
    print(load_train_csv().head())
    print(load_test_csv().head())
