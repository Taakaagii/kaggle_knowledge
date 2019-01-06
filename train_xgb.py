import pandas as pd
import numpy as np
import load as ld
import os as os
from logging import StreamHandler,DEBUG,Formatter,FileHandler,getLogger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,ParameterGrid
from sklearn.metrics import log_loss,roc_auc_score,roc_curve,auc
import xgboost as xgb
import tqdm as tqdm

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
    return 'gini', gini(y, pred)

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
    #all_param = {'max_depth': [3],
    #             'learning_rate': [0.1],
    #             'min_child_weight': [3],
    #             'n_estimators': [10000],
    #             'colsample_bytree': [0.8],
    #             'colsample_bylevel': [0.8],
    #             'reg_alpha': [0.1],
    #             'max_delta_step': [0.1],
    #             'seed': [0],
    #            }
    
    all_param = {'seed': [0]}
    
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

            logger.info('trn_x:{} , trn_y{}'.format(trn_x.shape,trn_y.shape))
            logger.info('val_x:{} , val_y{}'.format(val_x.shape,val_y.shape))
            
            #xgboost fitting
            clf = xgb.sklearn.XGBClassifier(**params)
            logger.info('model making end')
            clf.fit(trn_x,trn_y)
            
           #clf.fit(trn_x,
           #        trn_y,
           #        eval_set=[(val_x, val_y)],
           #        early_stopping_rounds=100,
           #        eval_metric=gini_xgb
           #        )
            
            logger.info('fitting end')
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
    clf = xgb.sklearn.XGBClassifier(**max_params)
    clf.fit(x_train,y_train)
    pred_test = clf.predict_proba(x_test)[:,1]

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['target'] = pred_test
    df_submit.to_csv(os.path.join(DIR,'sample.csv'),index=False)

    logger.info('all end')
