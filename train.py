import pandas as pd
import numpy as np
import load as ld
import os as os
from logging import StreamHandler,DEBUG,Formatter,FileHandler,getLogger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss,roc_auc_score,auc

logger = getLogger(__name__)
DIR = 'log/'
SAMPLE_SUBMIT_FILE = './input/sample_submission.csv'

if __name__ == '__main__':
    
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)
    
    handler = FileHandler(os.path.join(DIR,'train.py.log'),'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    
    logger.info('start')
    
    df = ld.load_train_csv()
    
    x_train = df.drop('target',axis=1)
    y_train = df['target']
    
    use_cols = x_train.columns.values
    logger.debug('train columns {} {}'.format(use_cols.shape,use_cols))
    logger.debug('train matrrix {}'.format(x_train.shape))
    
    logger.info('preparation end')
    
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    
    #cross validation
    for train_idx,valid_idx in cv.split(x_train,y_train):
        
        trn_x = x_train.iloc[train_idx,:]
        val_x = x_train.iloc[valid_idx,:]
        
        trn_y = y_train[train_idx]
        val_y = y_train[valid_idx]
        
        #logstic regression fit
        clf = LogisticRegression(random_state=42)
        clf.fit(trn_x.values,trn_y.values)
        pred = clf.predict_proba(val_x)[:,1]
        sc_logloss = log_loss(val_y,pred)
        sc_auc = roc_auc_score(val_y,pred)
        
        logger.info('logloss: {},auc: {}'.format(sc_logloss,sc_auc))
        
    logger.info('train end')
    
    df = ld.load_test_csv()

    x_test = df[use_cols].sort_values('id')

    logger.debug('test matrrix {}'.format(x_test.shape))
    pred_test = clf.predict_proba(x_test.values)

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['target'] = pred_test
    df_submit.to_csv(os.path.join(DIR,'sample.csv'),index=False)

    logger.info('all end')    
