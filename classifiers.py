import pandas as pd
import numpy as np
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, LabelEncoder
from tqdm import tqdm
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
train_df = pd.read_csv("train.csv")
strs = train_df['target'].value_counts()
value_map = dict(('Class'+'_'+str(i+1), i) for i in range(len(list(strs.index))))

train_df = train_df.replace({'target':value_map})
train_df = train_df.drop(columns=['id'])
features = [c for c in train_df if c not in ['target']]

test_df = pd.read_csv("test.csv")

### New Feature
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
enc_list = list(set(features)-set(['feature_45','feature_57','feature_27','feature_52','feature_47','feature_7','feature_66','feature_42','feature_49']))
for f in tqdm(enc_list):
    train_df[f + '_target_enc'] = 0
    test_df[f + '_target_enc'] = 0
    for i, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['target'])):
        trn_x = train_df[[f, 'target']].iloc[trn_idx].reset_index(drop=True)
        val_x = train_df[[f]].iloc[val_idx].reset_index(drop=True)
        enc_df = trn_x.groupby(f, as_index=False)['target'].agg({f + '_target_enc': 'mean'})
        val_x = val_x.merge(enc_df, on=f, how='left')
        test_x = test_df[[f]].merge(enc_df, on=f, how='left')
        val_x[f + '_target_enc'] = val_x[f + '_target_enc'].fillna(train_df['target'].mean())
        test_x[f + '_target_enc'] = test_x[f + '_target_enc'].fillna(train_df['target'].mean())
        train_df.loc[val_idx, f + '_target_enc'] = val_x[f + '_target_enc'].values
        test_df[f + '_target_enc'] += test_x[f + '_target_enc'].values / skf.n_splits

features = [c for c in train_df if c not in ['target']]
features = list(set(features)-set(['feature_45','feature_57','feature_27','feature_52','feature_47','feature_7','feature_66','feature_42','feature_49']))
target = train_df['target']

## Sampling
target = train_df['target'].values
class_sample_count = np.array(
    [len(np.where(target == t)[0]) for t in np.unique(target)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in target])

samples_weight = torch.from_numpy(samples_weight)
samples_weigth = samples_weight.double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

target = torch.from_numpy(target).long()
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_df.values).long(), target)

train_loader = DataLoader(
    train_dataset, batch_size=len(train_df), num_workers=1, sampler=sampler)
X = next(iter(train_loader))[0].numpy()
# target = X[:,-1]
target = X[:,-1]
X = X[:,:-1]
print(X.shape, target.shape)


#XGBOOST

# target = train_df['target']
cv_params = {'max_depth':[5,7,8,9,10,15,20,25]}#,'min_child_weight':[1,3,5,7,9,20,30,40,50] }

xgb_params = {'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'eta': 0.01,
            'gamma': 6,
            'subsample': 0.765,
            'grow_policy': 'depthwise', 
            'colsample_bytree': 0.512,
            'colsample_bylevel': 0.531,
            'colsample_bylevel': 0.7435,
            'lambda': 6.023,
            'alpha': 0.1472,
            'max_depth': 23,
            'min_child_weight': 255,
            'num_class':9,
            'max_bin': 366,
            'deterministic_histogram': False
            }

num_round = 2000
models = []
# for k,v in cv_params.items():
#     print('--------------- Cross Validation ** '+k+' ** --------------')
#     for pv in v:
#         xgb_params[k] = v
# folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=44000)
predictions = np.zeros((len(test_df), 9))
# for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
#     print("Fold {}".format(fold_))
#     dtrain = xgb.DMatrix(train_df.iloc[trn_idx][features], target.iloc[trn_idx])
#     dvalid = xgb.DMatrix(train_df.iloc[val_idx][features], target.iloc[val_idx])
#     evallist=[(dtrain,"train"),(dvalid,"valid")]

#     xgb_model = xgb.train(xgb_params, dtrain,evals=evallist, early_stopping_rounds=100,
#                        num_boost_round=100000)
#     predictions += xgb_model.predict(xgb.DMatrix(test_df[features]), ntree_limit=xgb_model.best_ntree_limit) / folds.n_splits

# X = train_df[features].copy()
X_test = test_df[features].copy()
# encoder = OneHotEncoder()
# all_encoded = encoder.fit_transform(X.append(X_test))
# #X = all_encoded[0:len(X)]
# #X_test = all_encoded[len(X):]
# X = all_encoded.tocsr()[0:len(X)]
# X_test = all_encoded [len(train_df):]

dtrain = xgb.DMatrix(X, target)
dvalid = xgb.DMatrix(X, target)
evallist=[(dtrain,"train"),(dvalid,"valid")]

xgb_model = xgb.train(xgb_params, dtrain,evals=evallist, early_stopping_rounds=100,verbose_eval=100,
                       num_boost_round=3600)

predictions += xgb_model.predict(xgb.DMatrix(X_test), ntree_limit=xgb_model.best_ntree_limit)
# fig, ax = plt.subplots(figsize=(14,28))
# xgb.plot_importance(xgb_model, height = 1, ax=ax)

# if os.path.isfile("feature_importance/xbg_3.png"):
#     os.remove("feature_importance/xbg_3.png")
# plt.savefig("feature_importance/xbg_3.png")
# fscore = xgb_model.get_score()
# s_fscore = {k:v for k,v in sorted(fscore.items(), key=lambda item:item[1],reverse=True)}
# s_fscore.keys()
# np.savetxt('feature_importance/xbg_3_keys.txt',list(s_fscore.keys()),fmt='%s')
        
outdict = {}
outdict['id'] = test_df['id']
for i in range(9):
    outdict['Class_'+str(i+1)] = predictions[:,i]
output = pd.DataFrame(outdict,index=None)
output.to_csv("result\\xgboost_with_tuning3.csv", index=False)

#GBDT
#Cross Validation 

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=44000)
oof = np.zeros((len(train_df), 9))
predictions = np.zeros((len(test_df), 9))
feature_importance_df = pd.DataFrame()

param = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 50,
    'num_class': 9,
    # 'max_depth':-1,
    'min_child_samples': 121,
    'max_bin': 15,
    'seed': 2021,
    'nthread': 6,
    'verbose': -1,
    'force_row_wise': True
        }

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
        print("Fold {}".format(fold_))
        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
        val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])
    
        num_round = 1000000
        clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
        # oof[val_idx] = clf.predict(df.iloc[val_idx][features], num_iteration=clf.best_iteration)
        
        # fold_importance_df = pd.DataFrame()
        # fold_importance_df["Feature"] = features
        # fold_importance_df["importance"] = clf.feature_importance()
        # fold_importance_df["fold"] = fold_ + 1
        # feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits
        
outdict = {}
outdict['id'] = test_df['id']
for i in range(9):
    outdict['Class_'+str(i+1)] = predictions[:,i]
output = pd.DataFrame(outdict,index=None)
output.to_csv('result\gbdt2_with_encodef.csv', index=False)