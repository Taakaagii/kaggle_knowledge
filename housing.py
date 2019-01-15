import pandas as pd
import time as time
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import Imputer as SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,ParameterGrid,train_test_split,cross_val_score,cross_validate
from sklearn.tree import tree,DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.svm import LinearSVR
from contextlib import contextmanager
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit

class Selector(BaseEstimator,TransformerMixin):
    def __init__(self,attr_name):
        self.attr_name = attr_name
    def fit(self,x,y=None):
        return self
    def transform(self,x):
        return x[self.attr_name]
    
class encoding(BaseEstimator,TransformerMixin):  
    def __init__(self,attr_name):
        self.attr_name = attr_name
    def fit(self,x,y=None):
        return self
    def transform(self,x):
        return pd.get_dummies(x[self.attr_name])
    
# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
    
with timer('csv preparation'):
    loaddata = pd.read_csv('./handson-ml-master/datasets/housing/housing.csv')
    
    # Divide by 1.5 to limit the number of income categories
    loaddata["income_cat"] = np.ceil(loaddata["median_income"] / 1.5)
    # Label those above 5 as 5
    loaddata["income_cat"].where(loaddata["income_cat"] < 5, 5.0, inplace=True)
    
    #train_set,test_set = train_test_split(loaddata,test_size=0.2,random_state=42)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    for train_index, test_index in split.split(loaddata,loaddata['income_cat']):
        strat_train_set = loaddata.loc[train_index]
        strat_test_set = loaddata.loc[test_index]
        
    for _set in (strat_train_set,strat_test_set):
        _set.drop("income_cat", axis=1, inplace=True)
        
    train_y = strat_train_set['median_house_value']
    train_x = strat_train_set.drop('median_house_value',axis = 1)
    test_y = strat_test_set['median_house_value']
    test_x = strat_test_set.drop('median_house_value',axis = 1)
    
    num_col = [x for x in train_x.columns.values if x != 'ocean_proximity']
    cat_col = ['ocean_proximity']
    
with timer('PipeLine'):
    num_pipe = Pipeline([
                        ('Selector',Selector(num_col)),
                        ('imputer',SimpleImputer(strategy='median')),
                        ('attribs_adder', CombinedAttributesAdder()),
                        ('std_scaler',StandardScaler())
                        ])
    
    cat_pipe = Pipeline([
                        ('Encoder',encoding(cat_col))
                        ])
    
    union = FeatureUnion(transformer_list=[('num_pipe',num_pipe),('cat_pipe',cat_pipe)])
    
    housing_x = union.fit_transform(train_x)
    housing_x_t = union.transform(test_x)

min_score = 2147483647
min_param = None
                
with timer('parameter search'):
    
    all_params = {'n_estimators': [3, 10, 30], 
                  'max_features': [2, 4, 6, 8],
                  'bootstrap': [False],
                  'random_state':[42]
                 }
    
    for param in tqdm(list(ParameterGrid(all_params))):
        with timer('cross validation'):
            clf = RandomForestRegressor(**param)
            scores = cross_val_score(clf,housing_x,train_y,scoring='neg_mean_squared_error',cv=5)
            # scores = cross_validation(clf,housing_x,train_y,scoring='neg_mean_squared_error',cv=10)
            tree_score = np.sqrt(-scores)
            
            print(np.mean(tree_score))
            if min_score > np.mean(tree_score):
                min_score = np.mean(tree_score)
                min_param = param

#print(min_score)
#print('param {}'.format(min_param))
clf = RandomForestRegressor(**min_param)
clf.fit(housing_x,train_y)
pred = clf.predict(housing_x_t)
score = np.sqrt(mean_squared_error(test_y,pred))
print(score)
