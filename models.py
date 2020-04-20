clf = RandomForestClassifier(random_state = 50, n_jobs = -1)
# Select best C for SVC by AUC score of ROC curve
best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,
                        param_grid={'n_estimators': [340,500,1000]})
best_clf.fit(train_set,train_label)
print("Select best RandomForest model with n_estimators = {} with best_score={}".format(
    best_clf.best_params_['n_estimators'],
    best_clf.best_score_))
test_clf=RandomForestClassifier(random_state = 1, n_jobs = -1,n_estimators=best_clf.best_params_['n_estimators'])
# best_clf.best_params_['n_estimators']
test_clf.fit(train_set,train_label)
print("The avarage AUC_ROC of the best random forest from 5-fold CV on test data is",
      roc_auc_score(y_true.tolist(),test_clf.predict(test_set).tolist()))
#%%
from sklearn.linear_model import LogisticRegression
# Make the model with the specified regularization parameter
clf = LogisticRegression()
best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,
                        param_grid={'C': [0.001]})
best_clf.fit(train_set,train_label)
print("Select best Logistic Regression model with C = {} with best_score={}".format(
    best_clf.best_params_['C'],
    best_clf.best_score_))
test_clf=LogisticRegression(C=best_clf.best_params_['C'])
test_clf.fit(train_set,train_label)
print("The avarage AUC_ROC of the best random forest from 5-fold CV on test data is",
      roc_auc_score(y_true.tolist(),test_clf.predict(test_set).tolist()))
#%%
for c in [0.001,0.01,0.1]:
    lg_clf=LogisticRegression(random_state = 1, n_jobs = -1,C=c)
    lg_clf.fit(train_set,train_label)
    print("The avarage AUC_ROC of the best model with {} from 5-fold CV on test data is".format(c),
        roc_auc_score(y_true.tolist(),lg_clf.predict(test_set).tolist()))
#%%
from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(hidden_layer_sizes=(30,10,5),random_state=50,activation='logistic')
best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,
                        param_grid={'alpha': [1.2,1.4,2]})
best_clf.fit(train_set,train_label)
print("Select best neural network model with alpha = {} with best_score={}".format(
    best_clf.best_params_['alpha'],
    best_clf.best_score_))

nn_clf=MLPClassifier(random_state = 50,activation='logistic',alpha=best_clf.best_params_['alpha'])
nn_clf.fit(train_set,train_label)
print("The avarage AUC_ROC of the best nn from 5-fold CV on test data is",
      roc_auc_score(y_true.tolist(),nn_clf.predict(test_set).tolist()))
#%%
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(weights='distance',n_jobs=-1)
best_clf = GridSearchCV(knn,scoring='roc_auc',cv=5,n_jobs=-1,
                        param_grid={'n_neighbors': [350,400,450,500]})

best_clf.fit(train_set,train_label)
print("Select best KNN model with n_neighbors = {} with best_score={}".format(
    best_clf.best_params_['n_neighbors'],
    best_clf.best_score_))
knn_clf=KNeighborsClassifier(weights='distance',n_jobs=-1,n_neighbors=best_clf.best_params_['n_neighbors'])
knn_clf.fit(train_set,train_label)
print("The avarage AUC_ROC of the best knn from 5-fold CV on test data is",
      roc_auc_score(y_true.tolist(),knn_clf.predict(test_set).tolist()))


#%%
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(fit_intercept=True,class_weight='balanced',max_iter=500,warm_start=True)
best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,
                        param_grid={'C': [0.00001,0.00005,0.0001],'penalty' : ['l2','l1'],})
best_clf.fit(train_pca,train_label)
print("Select best Logistic Regression model with C = {} and penalty {} with best_score={}".format(
    best_clf.best_params_['C'],best_clf.best_params_['penalty'],
    best_clf.best_score_))
#%%
lg_clf=LogisticRegression(fit_intercept=True,class_weight='balanced',max_iter=100,warm_start=True,C=0.065)
lg_clf.fit(train_pca,train_label)
print("The avarage AUC_ROC of the best random forest from 5-fold CV on test data is",
      roc_auc_score(y_true.tolist(),lg_clf.predict(test_pca).tolist()))
#%%
from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(hidden_layer_sizes=(3,2),random_state=50,activation='logistic',max_iter=500)
best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,
                        param_grid={'alpha': [1,2,3,4]})
best_clf.fit(train_pca,train_label)
print("Select best neural network model with alpha = {} with best_score={}".format(
    best_clf.best_params_['alpha'],
    best_clf.best_score_))
#%%
nn_clf=MLPClassifier(random_state=50,activation='logistic',alpha=best_clf.best_params_['alpha'])
nn_clf.fit(train_pca,train_label)
print("The avarage AUC_ROC of the best nn from 5-fold CV on test data is",
      roc_auc_score(y_true.tolist(),nn_clf.predict(test_pca).tolist()))

#%%
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(weights='distance',n_jobs=-1)
best_clf = GridSearchCV(knn,scoring='roc_auc',cv=5,n_jobs=-1,
                        param_grid={'n_neighbors': [480,500,550,600]})

best_clf.fit(train_pca,train_label)
print("Select best KNN model with n_neighbors = {} with best_score={}".format(
    best_clf.best_params_['n_neighbors'],
    best_clf.best_score_))
knn_clf=KNeighborsClassifier(weights='distance',n_jobs=-1,n_neighbors=best_clf.best_params_['n_neighbors'])
knn_clf.fit(train_pca,train_label)
print("The avarage AUC_ROC of the best knn from 5-fold CV on test data is",
      roc_auc_score(y_true.tolist(),knn_clf.predict(test_pca).tolist()))

#%%
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import GridSearchCV
clf = LGBMClassifier(random_state = 50, n_jobs = -1,reg_lambda=0.0095)
best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,
                        param_grid={'n_estimators': [600,650,700,750,800]})
best_clf.fit(train_pca,train_label)
print("Select best LGB model with n_estimators = {}  with best_score={}".format(
    best_clf.best_params_['n_estimators'],
    best_clf.best_score_))
LBMclf= LGBMClassifier(random_state = 50, n_jobs = -1,n_estimators=best_clf.best_params_['n_estimators'],reg_lambda=0.0095)
LBMclf.fit(train_pca,train_label)
print("The avarage AUC_ROC of the best lightGBM from 5-fold CV on test data is",
      roc_auc_score(y_true.tolist(),LBMclf.predict(test_pca).tolist()))


# %%
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
clf=XGBClassifier(random_state=50,n_jobs=-1,n_estimators=340)
best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,
                        param_grid={'reg_lambda':[0.001,0.01,0.1]})
best_clf.fit(train_set,train_label)
print("Select best xgboost model with reg_lambda = {} with best_score={}".format(
    best_clf.best_params_['reg_lambda'],
    best_clf.best_score_))
xgb_clf=XGBClassifier(n_jobs=-1,random_state=50,n_estimators=340,reg_lambda=best_clf.best_params_['reg_lambda'])
xgb_clf.fit(train_pca,train_label)
print("The avarage AUC_ROC of the best knn from 5-fold CV on test data is",
      roc_auc_score(y_true.tolist(),xgb_clf.predict(test_pca).tolist()))
# %%


# %%
from sklearn.svm import SVC
clf=SVC(kernel="rbf", degree=2, coef0=1,C=0.001,)
best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,
                        param_grid={'gamma':[1,3,5]})
best_clf.fit(train_pca,train_label)
print("Select best svc model with gamma = {} with best_score={}".format(
    best_clf.best_params_['gamma'],
    best_clf.best_score_))
clf=SVC(kernel="rbf", degree=2, coef0=1,C=0.001,gamma=best_clf.best_params_['gamma'])
clf.fit(train_pca,train_label)
print("The avarage AUC_ROC of the best knn from 5-fold CV on test data is",
      roc_auc_score(y_true.tolist(),clf.predict(test_pca).tolist()))

# %%
