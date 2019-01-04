import pandas as pd
import numpy as np
import load as ld
import os as os
from logging import StreamHandler,DEBUG,Formatter,FileHandler,getLogger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,ParameterGrid
from sklearn.metrics import log_loss,roc_auc_score,roc_curve,auc

logger = getLogger(__name__)
DIR = 'log/'
SAMPLE_SUBMIT_FILE = './input/sample_submission.csv'

# custom objective function (similar to auc)
def gini(y, pred):
    #g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    #g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    #gs = g[:,0].cumsum().sum() / g[:,0].sum()
    #gs -= (len(y) + 1) / 2.
    #return gs / len(y)
    fpr,tpr,thr = roc_curve(y,pred,pos_label=1)
    g = 2 * auc(fpr,tpr) - 1
    return g

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred) / gini(y, y)

def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True

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
    
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
    
    #Parameter grid search
    all_param ={'C':[10**i for i in range(-1,2)],
                'fit_intercept':[True,False],
                'penalty':['l2','l1'],
                'random_state':[0]}
    
    max_score = -100
    max_params = None
    
    for params in ParameterGrid(all_param):
        
        logger.info('params:{}'.format(params))
        
        #list_auc =[]
        list_gini =[]
        list_logloss =[]
        #cross validation
        for train_idx,valid_idx in cv.split(x_train,y_train):

            trn_x = x_train.iloc[train_idx,:]
            val_x = x_train.iloc[valid_idx,:]

            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]

            #logstic regression fit
            clf = LogisticRegression(**params)
            clf.fit(trn_x.values,trn_y.values)
            pred = clf.predict_proba(val_x)[:,1]
            sc_logloss = log_loss(val_y,pred)
            #sc_auc = roc_auc_score(val_y,pred)
            sc_gini = gini(val_y,pred)

            #list_auc.append(sc_auc)
            list_gini.append(sc_gini)
            list_logloss.append(sc_logloss)
            #logger.info('logloss: {},auc: {}'.format(sc_logloss,sc_auc))
            logger.info('logloss: {},gini: {}'.format(sc_logloss,sc_gini))
            break
            
        #if max_score < np.mean(list_auc):
        #    max_score = list_auc
        #    max_params = params
        
        if max_score < np.mean(list_gini):
            max_score = np.mean(list_gini)
            max_params = params
            
    #logger.info('max_auc: {} max_params: {}'.format(max_score,max_params))
    logger.info('max_gini: {} max_params: {}'.format(max_score,max_params))    
    logger.info('train end')
    
    df = ld.load_test_csv()

    x_test = df[use_cols].sort_values('id')

    logger.debug('test matrrix {}'.format(x_test.shape))
    clf = LogisticRegression(**max_params)
    clf.fit(x_train.values,y_train.values)
    pred_test = clf.predict_proba(x_test.values)

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['target'] = pred_test
    df_submit.to_csv(os.path.join(DIR,'sample.csv'),index=False)

    logger.info('all end')   
