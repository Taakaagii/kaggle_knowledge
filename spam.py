import os
import tarfile
from six.moves import urllib
import numpy as np

DOWN_ROOT = 'https://spamassassin.apache.org/old/publiccorpus/'
HAM_URL = DOWN_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWN_ROOT + "20030228_spam.tar.bz2" 
SPAM_PATH = os.path.join("./datasets", "spam")
ham_files = []
spam_files =[]

def spam_fetch():
    
    if not os.path.isdir('./datasets'):
        os.mkdir('./datasets')
        
    if not os.path.isdir(SPAM_PATH):
        os.mkdir(SPAM_PATH)
        
    for filename,url in (('ham.tar.bz2',HAM_URL),('spam.tar.bz2',SPAM_URL)):
        path = os.path.join(SPAM_PATH,filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url,path)
            
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=SPAM_PATH)
        tar_bz2_file.close()

def list_name():
    
    easy_ham = os.path.join(SPAM_PATH,'easy_ham')
    spam = os.path.join(SPAM_PATH,'spam')
    
    global ham_files
    global spam_files
    
    ham_files = [name for name in os.listdir(easy_ham) if len(name) > 20]
    spam_files = [name for name in os.listdir(spam) if len(name) > 20]
    
    print(len(ham_files),len(spam_files))

import email
import email.policy

def load_email(is_spam, filename):
    
    directory = "spam" if is_spam else "easy_ham" 
    
    with open(os.path.join(SPAM_PATH, directory, filename), "rb") as f: 
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

#print(ham_emails[200])

def get_email_struct(email):
    
    if isinstance(email,str):
        return email
    payload = email.get_payload()
    
    if isinstance(payload,list):
        return 'multiple({})'.format(','.join([get_email_struct(sub_email) for sub_email in payload]))
    else:
        return email.get_content_type()

def count_email_type(emails):
    dic = {}
    for email in emails:
        key = get_email_struct(email)
        if key in dic:
            dic[key] +=1
        else:
            dic[key] = 1
    return dic

import matplotlib.pyplot as plt
%matplotlib inline

def plot_emaildata():
    ham_dic = sorted(count_email_type(ham_emails).items(), key=lambda x: -x[1])
    spam_dic = sorted(count_email_type(spam_emails).items(), key=lambda x: -x[1])
    ham_type_key ,ham_type_val= [k for k,v in ham_dic],[v for k,v in ham_dic]
    spam_type_key,spam_type_val = [k for k,v in spam_dic],[v for k,v in spam_dic]
    fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(15,8))
    ax1.bar(np.arange(len(ham_type_key)),height=ham_type_val,tick_label=ham_type_key)
    ax1.tick_params(rotation=90)
    ax1.set_title("HAM")
    ax2.bar(np.arange(len(spam_type_key)),height=spam_type_val,tick_label=spam_type_key)
    ax2.tick_params(rotation=90)
    ax2.set_title("SPAM")
    
from sklearn.model_selection import train_test_split

def split_train_test(ham,spam):
    
    all_x = np.array(ham + spam)
    all_y = np.array(len(ham) * [0] + len(spam) * [1])
    
    return train_test_split(all_x,all_y,test_size=0.2,random_state=42)

import re
from html import unescape

#html変換用メソッド
def html_to_plain_text(html): 
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I) 
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)

def Totext(email):
    html = None
    for x in email.walk():
        type_d = x.get_content_type()
        if not type_d in ['text/plain','text/html']:
            continue
        try:
            content = x.get_content()
        except:
            content = str(x.get_payload())
        #return していいの？
        if type_d == 'text/plain':
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(content)
    
from sklearn.base import BaseEstimator,TransformerMixin
import urlextract
from collections import Counter
import nltk

url_extractor = urlextract.URLExtract()
stemmer = nltk.PorterStemmer()

class ToWordTransformer(BaseEstimator,TransformerMixin):
    
    def __init__(self,strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
        
    def fit(self,x,y=None):
        return self
    
    def transform(self,x,y=None):
        x_transformed = []
        for email in x:
            
            text = Totext(email) or ""
            if self.lower_case:
                text = text.lower()
            
            if self.replace_urls:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)
                
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
                
            word_counts = Counter(text.split())    
                
            if self.stemming:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
                
            x_transformed.append(word_counts)
            
        return np.array(x_transformed)
                
        #    if self.stemming:
        #        stemmed_word_counts = []
        #        for word, count in word_counts.items():
        #            stemmed_word = stemmer.stem(word)
        #            #stemmed_word_counts[stemmed_word] += count
        #            stemmed_word_counts.append(stemmed_word)
        #        word_counts= u" ".join(stemmed_word_counts[1:-1])
        #        
        #    X_transformed.append(word_counts)
        #return X_transformed
    
from scipy.sparse import csr_matrix
    
class WordToSparse(BaseEstimator,TransformerMixin):
    
    def __init__(self,voc_size=1000):
        self.voc_size = voc_size
    
    def fit(self, x, y=None):
        total_count = Counter()
        for word_count in x:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.voc_size]
        self.most_common_ = most_common
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, x, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(x):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(x), self.voc_size + 1))
        
        
#spam_fetch()
list_name()
ham_emails = [load_email(is_spam=False, filename=name) for name in ham_files]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_files]
#plot_emaildata()
train_x,test_x,train_y,test_y=split_train_test(ham_emails,spam_emails)

#x_few = train_x[:3]
#x_few_wordcounts = ToWordTransformer().fit_transform(x_few)
#x_few_vectors = WordToSparse(voc_size=10).fit_transform(x_few_wordcounts)
#x_few_vectors.toarray()

#sklearnを使用した場合
#from sklearn.feature_extraction.text import CountVectorizer
#count_vectorizer = CountVectorizer(max_features=10)
#feature_vectors = count_vectorizer.fit_transform(x_few_wordcounts)  # csr_matrix(疎行列)が返る
#feature_vectors.toarray()

from sklearn.pipeline import Pipeline

#countVectorizerをWordToSparseの代わりに試す
preprocess_pipeline = Pipeline([
    ("ToWordTransformer", ToWordTransformer()),
    ("WordToSparse", WordToSparse(voc_size=100)),
])

train_transformed_x = preprocess_pipeline.fit_transform(train_x)
test_transformed_x = preprocess_pipeline.transform(test_x)

#from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score,accuracy_score
from sklearn.metrics import roc_curve, auc
#
#cv_log = LogisticRegressionCV()
#cv_log.fit(train_transformed_x,train_y)
#
#ml_log = LogisticRegression(C=cv_log.C_[0])#best parameter
#ml_log.fit(train_transformed_x,train_y)
#
#score_log = cross_val_score(ml_log, train_transformed_x, train_y, cv=3, verbose=3)
#
#pred_log_y = ml_log.predict(test_transformed_x)
#fpr_log, tpr_log, _ = roc_curve(test_y,pred_log_y)
#
#print("CV mean:       {}".format(score_log.mean()))
#print("CV sd:         {}".format(np.sqrt(np.var(score_log))))
#print("Test accuracy: {}".format(ml_log.score(test_transformed_x,test_y)))
#print("Precision:     {}".format(precision_score(test_y,pred_log_y)))
#print("Recall:        {}".format(recall_score(test_y,pred_log_y)))
#print("AUC:           {}".format(auc(fpr_log,tpr_log)))
#
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import cross_val_score
#
#log_clf = LogisticRegression(solver="liblinear", random_state=42)
#score = cross_val_score(log_clf, train_transformed_x, train_y, cv=3, verbose=3)
#score.mean()
#
#from sklearn.metrics import precision_score, recall_score
#
#test_transformed_x = preprocess_pipeline.transform(test_x)
#
#log_clf = LogisticRegression(solver="liblinear", random_state=42)
#log_clf.fit(train_transformed_x, train_y)
#
#pred_y = log_clf.predict(test_transformed_x)
#
#print("Precision: {:.2f}%".format(100 * precision_score(test_y, pred_y)))
#print("Recall: {:.2f}%".format(100 * recall_score(test_y, pred_y)))

#xgboost のパラメーターチューニング等を試す

import xgboost as xgb
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from sklearn.model_selection import KFold

all_param = {
             #'max_depth': [8,9,10],
             #'learning_rate': [0.1],
             #'min_child_weight': [10],
             #'n_estimators': [500],
             #'colsample_bytree': [0.9],
             #'colsample_bylevel': [0.9],
             #'reg_alpha': [0,0.1],
             #'max_delta_step': [0,0.1],
             'seed': [42]
            }
#max_score = 0
#max_param = None

#kfold = KFold(n_splits=3,random_state=42)
#
#for trn_idx,val_idx in kfold.split(train_transformed_x,train_y):
#    for param in tqdm(list(ParameterGrid(all_param))):
#        clf = xgb.sklearn.XGBClassifier(**param)
#        #score = cross_val_score(clf,train_x,train_y,scoring='',cv=3)
#        trn_x = train_transformed_x[trn_idx]
#        trn_y = train_y[trn_idx]
#        val_x = train_transformed_x[val_idx]
#        val_y = train_y[val_idx]
#        
#        clf.fit(trn_x,
#                trn_y,
#                eval_set=[(val_x,val_y)],
#                #eval_metric='auc',
#                #early_stopping_rounds=200
#               )
#        
#        pred_y = clf.predict(val_x)
#        
#        if accuracy_score(pred_y,val_y) > max_score:
#            max_score = accuracy_score(pred_y,val_y)
#            max_param = param
#            
#    break
clf = xgb.sklearn.XGBClassifier()
clf.fit(train_transformed_x,train_y)
score = cross_val_score(clf,train_transformed_x,train_y,cv=3,verbose=3)

#clf = xgb.sklearn.XGBClassifier(**max_param)
#clf.fit(train_transformed_x,train_y)
test_pred = clf.predict(test_transformed_x)

print(np.mean(score))
print(accuracy_score(test_y,test_pred))
            

