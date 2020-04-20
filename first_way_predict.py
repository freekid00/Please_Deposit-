#%%
import pandas as pd 
import numpy as np 
import os
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
class campaign_age_unknown_trans(BaseEstimator,TransformerMixin):
    def fit(self,df,y=None):
        return self
    def transform(self,df):
        df=df.replace(to_replace={"unknown":np.nan})
        df.loc[(df['age']<0)|(df['age']>100),'age'] =  np.nan
        df['marital'] = df['marital'].replace(to_replace={"sungle":"single"})
        df.loc[(df['pdays'] ==999) & (df['poutcome'] !='nonexistent'),'pdays'] = np.nan
        q1,q3 = df['campaign'].quantile([0.25,0.75])
        lower = q1 - 3*(q3-q1)
        upper = q3 + 3*(q3-q1)
        df.loc[(df['campaign']<lower)|(df['campaign']>upper),'campaign'] = np.nan
        df.set_index('previous',inplace=True)
        df['campaign'] = df['campaign'].interpolate('linear')
        df.reset_index(inplace=True)
        df = df.assign(contacts_daily=(df['campaign']/(df['pdays']+1)).values)
        df[(df['age']>=60)&(pd.isnull(df.job))]['job']='retired'
        return df

class fix_imbalance(BaseEstimator,TransformerMixin):
    def fit(self,df,y=None):
        return self 
    def transform(self,df):
        self.class_priors_pos = (df['y']  == 'yes').sum()
        self.class_priors_neg = (df['y']  == 'no').sum()
        self.df_pos = df[df['y'] == 'yes']
        self.df_neg = df[df['y']  == 'no']
        self.df_pos_over = self.df_pos.sample(int(0.5*self.class_priors_neg), replace=True)
        df = pd.concat([self.df_pos_over,self.df_neg])
        return df

class encode_mix_std(BaseEstimator,TransformerMixin):
    def fit(self,df,y=None):
        return self 
    def transform(self,df):
        num_list=['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','contacts_daily']
        num_group=df[num_list]
        num_imp=SimpleImputer(strategy='mean')
        num_group=num_imp.fit_transform(num_group)
        std_=StandardScaler()
        num_group=std_.fit_transform(num_group)
        df=df.drop(num_list,axis=1)
        df['y']=df['y'].replace({'yes':True,'no':False})
        encoder_columns=list(df.columns)[:-1]
        one_hot_group=df[encoder_columns]
        df=df.drop(encoder_columns,axis=1)
        imp=SimpleImputer(strategy='most_frequent')
        one_hot_group=imp.fit_transform(one_hot_group)
        one_hot=OneHotEncoder(handle_unknown='ignore')
        one_hot_group=one_hot.fit_transform(one_hot_group)
        one_hot_group=one_hot_group.toarray()
        train_label=df['y']
        total_data=np.concatenate([num_group,one_hot_group],axis=1)
        return total_data,train_label


handle_pipeline = Pipeline([
        ('step1', campaign_age_unknown_trans()),
        # ('step2', assign_educ_job_marital()),
        ('step2', fix_imbalance()),
        ('step3',encode_mix_std())
    ])

strat_train_set=pd.read_csv('./data/strat_train_set.csv',index_col=False)
strat_test_set=pd.read_csv('./data/strat_test_set.csv',index_col=False)

train_set, train_label=handle_pipeline.fit_transform(strat_train_set)


handle_test_pipeline = Pipeline([
        ('step1', campaign_age_unknown_trans()),
        ('step2',encode_mix_std())
    ])
test_set,test_label=handle_test_pipeline.fit_transform(strat_test_set)
y_true=test_label
#%%
clf = RandomForestClassifier(random_state = 42, n_jobs = -1)
best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,
                        param_grid={'n_estimators': [100,200,500]})
best_clf.fit(train_set,train_label)
print("Select best RandomForest model with n_estimators = {} with best_score={}".format(
    best_clf.best_params_['n_estimators'],
    best_clf.best_score_))
for n in [1000,1500]:
    test_clf=RandomForestClassifier(random_state =42, n_jobs = -1,n_estimators=n)
    test_clf.fit(train_set,train_label)
    print("The avarage AUC_ROC of the best random forest from 5-fold CV on test data is",
        roc_auc_score(y_true.tolist(),test_clf.predict(test_set).tolist()))
#%%
clf = LogisticRegression()
best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,
                        param_grid={'C': [0.001,0.01,0.1,1,10,100]})
best_clf.fit(train_set,train_label)
print("Select best Logistic Regression model with C = {} with best_score={}".format(
    best_clf.best_params_['C'],
    best_clf.best_score_))

for c in  [0.001,0.01,0.1,1,10,100]:
    test_clf=LogisticRegression(C=c)
    test_clf.fit(train_set,train_label)
    print("The avarage AUC_ROC of the best logistic regression with {} from 5-fold CV on test data is".format(c),
        roc_auc_score(y_true.tolist(),test_clf.predict(test_set).tolist()))
#%%
clf=MLPClassifier(hidden_layer_sizes=(30,10),random_state=42,activation='logistic')
best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,
                        param_grid={'alpha': [0.02,0.05,0.07]})
best_clf.fit(train_set,train_label)
print("Select best neural network model with alpha = {} with best_score={}".format(
    best_clf.best_params_['alpha'],
    best_clf.best_score_))
#%%
for a in [0.05,0.08,0.1,0.12,0.5]:
    nn_clf=MLPClassifier(random_state = 42,activation='logistic',alpha=a,hidden_layer_sizes=(30,10))
    nn_clf.fit(train_set,train_label)
    print("The avarage AUC_ROC of the best nn and alpha={} from 5-fold CV on test data is".format(a),
        roc_auc_score(y_true.tolist(),nn_clf.predict(test_set).tolist()))
# %%
train=pd.read_csv('./data/bank_marketing_train.csv')
test=pd.read_csv('./data/bank_marketing_test.csv')

handle_pipeline = Pipeline([
        ('step1', campaign_age_unknown_trans()),
        ('step2', fix_imbalance()),
    ])
change1_test=campaign_age_unknown_trans()
train=handle_pipeline.fit_transform(train)
test=change1_test.fit_transform(test)

num_list=['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','contacts_daily']
train_num_group=train[num_list]
test_num_group=test[num_list]
num_imp=SimpleImputer(strategy='mean')
train_num_group=num_imp.fit_transform(train_num_group)
test_num_group=num_imp.fit_transform(test_num_group)
std_=StandardScaler()
train_num_group=std_.fit_transform(train_num_group)
test_num_group=std_.fit_transform(test_num_group)

train=train.drop(num_list,axis=1)
test=test.drop(num_list,axis=1)
encoder_columns=list(train.columns)[:-1]
one_hot_train=train[encoder_columns]
imp=SimpleImputer(strategy='most_frequent')
one_hot_train=imp.fit_transform(one_hot_train)
one_hot_test=imp.transform(test)
one_hot=OneHotEncoder(handle_unknown='ignore')
one_hot_train=one_hot.fit_transform(one_hot_train)
one_hot_test=one_hot.transform(one_hot_test)
one_hot_train=one_hot_train.toarray()
one_hot_test=one_hot_test.toarray()
train_data=np.concatenate([train_num_group,one_hot_train],axis=1)
test_data=np.concatenate([test_num_group,one_hot_test],axis=1)
train_label=train['y'].replace({'yes':True,'no':False})


# %%
clf = LogisticRegression()
best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,
                        param_grid={'C': [0.001,0.01,0.1,1,10,100]})
best_clf.fit(train_data,train_label)
print("Select best Logistic Regression model with C = {} with best_score={}".format(
    best_clf.best_params_['C'],
    best_clf.best_score_))

# %%
clf = LogisticRegression(C=1)
clf.fit(train_data,train_label)
test_ans=clf.predict_proba(test_data)
ans=pd.DataFrame(test_ans,columns=['prob_False','prob_True'])
ans.to_csv('./data/ans.csv')
test_label_lr=clf.predict(test_data)
test_label_lr=pd.DataFrame(test_label_lr,columns=['label'])
test_label_lr.to_csv('./data/testlabel_lr.csv')
# %%
