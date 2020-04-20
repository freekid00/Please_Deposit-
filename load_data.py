#%%
import pandas as pd 
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import os

train=pd.read_csv('./data/bank_marketing_train.csv')
train['y'].value_counts()/len(train)

### distribution of y is No:0.88, Yes:0.11

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(train, train["y"]):
    strat_train_set = train.loc[train_index]
    strat_test_set = train.loc[test_index]
print('shape of strat train',strat_train_set.shape)
print('shape of strat test',strat_test_set.shape)
strat_train_set.to_csv('./data/strat_train_set.csv',index=False)
strat_test_set.to_csv('./data/strat_test_set.csv',index=False)
#%%
import seaborn as sns 
def plot_kde(col,df,title_name,x_label):
    plt.figure()
    sns.kdeplot(df[df.y=='yes'][col],color='red',label='y==yes')
    sns.kdeplot(df[df.y=='no'][col],color='blue',label='y==no')
    plt.xlabel(x_label)
    plt.title(title_name)
    plt.legend()
#%%
## handle numeric datasets
num_list=['age','campaign','pdays','previous','emp.var.rate',
'cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y']
train_num=strat_train_set[num_list]
train_num.hist(bins=50, figsize=(8,8))


### Handle age
print('max of age',max(train_num['age']))
print('min of age',min(train_num['age']))
plot_kde('age',train_num,'Age kde plot','Age')

#%%
age_data=train_num[['age','y']]
age_data['age_bined']=pd.cut(age_data['age'],bins=np.linspace(15,100,num=18))
age_data=age_data.replace(to_replace={'yes':True,'no':False})
age_groups=age_data.groupby('age_bined').mean()
plt.figure(figsize = (4, 4))
# Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.index.astype(str), 100 * age_groups['y'])
# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Willingness to subscribe (%)')
plt.title('Failure to Repay by Age Group');

#%%
### Handle campaign
print('max of campaign',max(train_num['campaign']))
print('min of campaign',min(train_num['campaign']))
print('before handling campaign')
plot_kde('campaign',train_num,'Campaign kde plot','Campaign')
q1,q3 = train_num['campaign'].quantile([0.25,0.75])
lower = q1 - 3*(q3-q1)
upper = q3 + 3*(q3-q1)
train_num.loc[(train_num['campaign']<lower)|(train_num['campaign']>upper),'campaign'] = np.nan
print('after handling campaign')
plot_kde('campaign',train_num,'Campaign kde plot','Campaign')


### Handle cons.conf.idx
print('max of cons.conf.idx',max(train_num['cons.conf.idx']))
print('min of cons.conf.idx',min(train_num['cons.conf.idx']))
print('before handling cons.conf.idx')
plot_kde('cons.conf.idx',train_num,'cons.conf.idx kde plot','cons.conf.idx')

### Handle cons.price.idx
print('max of cons.price.idx',max(train_num['cons.price.idx']))
print('min of cons.price.idx',min(train_num['cons.price.idx']))
print('before handling cons.price.idx')
plot_kde('cons.price.idx',train_num,'cons.price.idx kde plot','cons.price.idx')

### Handle emp.var.rate
print('max of emp.var.rate',max(train_num['emp.var.rate']))
print('min of emp.var.rate',min(train_num['emp.var.rate']))
print('before handling emp.var.rate')
plot_kde('emp.var.rate',train_num,'emp.var.rate kde plot','emp.var.rate')

### Handle euribor3m
print('max of euribor3m',max(train_num['euribor3m']))
print('min of euribor3m',min(train_num['euribor3m']))
print('before handling euribor3m')
plot_kde('euribor3m',train_num,'euribor3m kde plot','euribor3m')

### Handle nr.employed
print('max of nr.employed',max(train_num['nr.employed']))
print('min of nr.employed',min(train_num['nr.employed']))
print('before handling nr.employed')
plot_kde('nr.employed',train_num,'nr.employed kde plot','nr.employed')

### Handle pdays
print('max of pdays',max(train_num['pdays']))
print('min of pdays',min(train_num['pdays']))
print('before handling pdays')
train_num.pdays.unique()

### Handle previous
print('max of previous',max(train_num['previous']))
print('min of previous',min(train_num['previous']))
print('before handling previous')
train_num.previous.unique()

#%%
from sklearn.base import BaseEstimator,TransformerMixin
class campaign_trans(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,df):
        q1,q3 = df['campaign'].quantile([0.25,0.75])
        lower = q1 - 3*(q3-q1)
        upper = q3 + 3*(q3-q1)
        df.loc[(df['campaign']<lower)|(df['campaign']>upper),'campaign'] = np.nan
        return df

strat_train_set=pd.read_csv('./data/strat_train_set.csv',index_col=False)
strat_test_set=pd.read_csv('./data/strat_test_set.csv',index_col=False)
cam_tran=campaign_trans()
strat_train_set=cam_tran.fit_transform(strat_train_set)
strat_test_set=cam_tran.fit_transform(strat_test_set)
#%%
obj_list=[]
for col in list(strat_train_set.columns):
    if strat_train_set[col].dtype=='object':
        obj_list.append(col)

train_obj=strat_train_set[obj_list].replace(to_replace={"unknown":np.nan})
#%%
train_obj.isnull().sum()/20996*100
#%%
print(strat_train_set.job.unique())
print(strat_train_set[train_obj['job'].isnull()].sort_values('age'))
train_obj[strat_train_set['age']>=60]['job']='retired'
#%%
print('after attribute retired')
print(train_obj.isnull().sum()/20996*100)
print(train_obj.job.value_counts()/len(train_obj))
#%%
print(train_obj['marital'].unique())
print(train_obj.marital.value_counts()/len(train_obj))
# %%
print(train_obj.education.unique())
print(train_obj.education.value_counts()/len(train_obj))
# %%
train_obj.loc[train_obj.job=='student','education']='high.school'
train_obj.loc[train_obj.job=='technician','education']='professional.course'
train_obj.loc[train_obj.job=='self-employed','education']='university.degree'
train_obj.loc[train_obj.job=='admin.','education']='university.degree'
train_obj.loc[train_obj.job=='entrepreneur','education']='university.degree'
train_obj.loc[train_obj.job=='housemaid','education']='basic.4y'



# %%
strat_train_set=pd.read_csv('./data/strat_train_set.csv',index_col=False)
strat_test_set=pd.read_csv('./data/strat_test_set.csv',index_col=False)
cam_tran=campaign_trans()
strat_train_set=cam_tran.fit_transform(strat_train_set)
strat_test_set=cam_tran.fit_transform(strat_test_set)
strat_train_set=strat_train_set.replace(to_replace={"unknown":np.nan})
strat_train_set[(strat_train_set['age']>=60)&(pd.isnull(strat_train_set.job))]['job']='retired'
print(strat_train_set.isnull().sum())
print(strat_train_set.shape)
def assign_education(df):
    edu_ass=df.education.value_counts().index[0]
    df.loc[pd.isnull(df.education),'education']=edu_ass
    return df 

def assign_job(df):
    df['job'] = df['job'].fillna(method='ffill')
    return df.reset_index(drop=True)

strat_train_set=strat_train_set.groupby('age').apply(assign_job).reset_index(drop=True)
strat_train_set=strat_train_set.groupby('job').apply(assign_education).reset_index(drop=True)
strat_train_set.loc[pd.isnull(strat_train_set.marital),'marital']='married'

print("#"*30)
print('Fix imbalanced data:')
class_priors_pos = (strat_train_set['y']  == 'yes').sum()
class_priors_neg = (strat_train_set['y']  == 'no').sum()
print("There are {} positive examples and {} negative examples in the data set".format(
        class_priors_pos,class_priors_neg))

df_pos = strat_train_set[strat_train_set['y'] == 'yes']
df_neg = strat_train_set[strat_train_set['y']  == 'no']
df_pos_over = df_pos.sample(int(0.8*class_priors_neg), replace=True)
strat_train_set = pd.concat([df_pos_over,df_neg])
class_priors_pos = (strat_train_set['y']  == 'yes').sum()
class_priors_neg = (strat_train_set['y']  == 'no').sum()
print("After over-sampling, there are {} positive examples and {} negative examples in the data set".format(
        class_priors_pos,class_priors_neg))
print(strat_train_set.isnull().sum())
print(strat_train_set.shape)

def obj_distribution(col_name,df):
    col_df=df[[col_name,'y']]
    col_df.y=col_df.y.replace({'yes':True,'no':False})
    col_df=col_df.groupby(col_name).mean().reset_index().sort_values('y',ascending=False)
    plt.bar(col_df[col_name].astype(str), 100 * col_df['y'])
    plt.xticks(rotation = 75)
    plt.title(col_name+' dist plot')
    plt.show()

obj_list=[]
for col in list(strat_train_set.columns):
    if strat_train_set[col].dtype=='object':
        obj_list.append(col)
obj_list=obj_list[:-1]
for obj_col in obj_list:
    obj_distribution(obj_col,strat_train_set)


from sklearn.preprocessing import OneHotEncoder
one_hot_groupname=['marital','day_of_week']
one_hot_group=strat_train_set[one_hot_groupname]
strat_train_set=strat_train_set.drop(one_hot_groupname,axis=1)
one_hot=OneHotEncoder()
one_hot_group=one_hot.fit_transform(one_hot_group)
one_hot_group=one_hot_group.toarray()

obj_list=[]
for col in list(strat_train_set.columns):
    if strat_train_set[col].dtype=='object':
        obj_list.append(col)
num_list=['age','campaign','pdays','previous','emp.var.rate',
'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
num_group=strat_train_set[num_list]
from sklearn.preprocessing import StandardScaler
std_=StandardScaler()
num_group=std_.fit_transform(num_group)
strat_train_set=strat_train_set.drop(num_list,axis=1)

class encoder_by_importance(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        X.y=X.y.replace({'yes':True,'no':False})
        return self
    def transform(self,X):
        encoder_columns=list(X.columns)[:-1]
        for col_name in encoder_columns:
            df=X[[col_name,'y']]
            df=df.groupby(col_name).mean().sort_values('y').reset_index()
            num=1
            match_dict=dict()
            for i in df.iloc[:,0]:
                match_dict[i]=num
                num+=1
            X[col_name]=X[col_name].replace(match_dict)
        return X

en_importance=encoder_by_importance()
en_data=en_importance.fit_transform(strat_train_set)
train_label=en_data['y']
en_data=en_data.drop('y',axis=1)

total_data=np.concatenate([num_group,one_hot_group,en_data.values],axis=1)

from sklearn.impute import KNNImputer
knn_imp=KNNImputer(n_neighbors=200)
total_data=knn_imp.fit_transform(total_data)

# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# clf = SVC()
# # Select best C for SVC by AUC score of ROC curve
# best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,
#                         param_grid={'C': [0.01, 0.1, 1, 10, 100,500,1000]})
# best_clf.fit(total_data,train_label)
# print("Select best SVC model with C = {} with best_score={}".format(
#     best_clf.best_params_['C'],
#     best_clf.best_score_))


# #%%
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# clf = RandomForestClassifier(random_state = 50, n_jobs = -1)
# # Select best C for SVC by AUC score of ROC curve
# best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,
#                         param_grid={'n_estimators': [330,340,350]})
# best_clf.fit(total_data,train_label)
# print("Select best RandomForest model with n_estimators = {} with best_score={}".format(
#     best_clf.best_params_['n_estimators'],
#     best_clf.best_score_))

# # %%
# from sklearn.linear_model import LogisticRegression
# # Make the model with the specified regularization parameter
# clf = LogisticRegression()
# best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,
#                         param_grid={'C': [0.068,0.069,0.07]})
# best_clf.fit(total_data,train_label)
# print("Select best Logistic Regression model with C = {} with best_score={}".format(
#     best_clf.best_params_['C'],
#     best_clf.best_score_))

# # %%
