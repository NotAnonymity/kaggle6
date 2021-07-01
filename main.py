import pandas as pd
import numpy as np
import random
import glob
import os
import gc
import re

from sklearn.metrics import log_loss, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K

from tensorflow.keras import layers
from tensorflow.keras import activations, callbacks
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import initializers
from tensorflow.keras.utils import plot_model

from tensorflow.keras import Model
import keras_tuner as kt
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from knn_feature import get_knn_feature

CFG = {
    'target': 'target',
    'n_class': 9,
    'seed': 2021,
    'k': 5,
    'n_clusters': 9,
    'n_components': 2,
    'emb_out_dim': 16,
    'max_epochs': 50,
    'batch_size': 256,
    'learning_rate': 1e-3,
    'es_patience': 5,
    'lr_patience': 2,
    'lr_factor': 0.7,
    'n_splits': 10,
    'verbose': 0,
}


def seed_everything(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(CFG['seed'])

SAVE_PATH = f'result/'
os.makedirs(SAVE_PATH, exist_ok=True)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")

target_ohe = pd.get_dummies(train[CFG['target']])
target = train[CFG['target']].apply(lambda x: int(x.split("_")[-1])-1)
features = [col for col in train.columns if col.startswith('feature_')]

scaler = MinMaxScaler()
all_df = pd.concat([train, test]).reset_index(drop=True)
all_df = scaler.fit_transform(all_df[features])
train_2d = all_df[:train.shape[0]].reshape(-1, 5, 5, 3)
test_2d = all_df[train.shape[0]:].reshape(-1, 5, 5, 3)

km = KMeans(n_clusters=CFG['n_clusters'], random_state=CFG['seed'])
all_km = km.fit_transform(all_df)
train_km = all_km[:train.shape[0]]
test_km = all_km[train.shape[0]:]

pca = PCA(n_components=CFG['n_components'], random_state=CFG['seed'])
all_pca = pca.fit_transform(all_df)
train_pca = all_pca[:train.shape[0]]
test_pca = all_pca[train.shape[0]:]

knn_train, knn_test = get_knn_feature(train, test)
all_knn = np.concatenate([
        knn_train,
        knn_test
        ])
all_knn = scaler.fit_transform(all_knn)
train_knn = all_knn[:train.shape[0]]
test_knn = all_knn[train.shape[0]:]

def custom_metric(y_true, y_pred):
    y_pred = K.clip(y_pred, 1e-15, 1-1e-15)
    loss = K.mean(cce(y_true, y_pred))
    return loss

cce = tf.keras.losses.CategoricalCrossentropy()

es = tf.keras.callbacks.EarlyStopping(
    monitor='val_custom_metric',
    min_delta=1e-05,
    patience=CFG['es_patience'],
    verbose=CFG['verbose'],
    mode='min',
    baseline=None,
    restore_best_weights=True)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_custom_metric',
    factor=CFG['lr_factor'],
    patience=CFG['lr_patience'],
    verbose=CFG['verbose'],
    mode='min')

emb_in_dim = pd.concat([train, test]).reset_index(drop=True)[features].max().max()+1
emb_out_dim = CFG['emb_out_dim']
emb_dims = [emb_in_dim, emb_out_dim]

def create_model(emb_dims):
    #--------------------------------------
    conv_inputs = layers.Input(shape=(75))
    conv2_inputs = layers.Input(shape=(5, 5, 3))
    knn_inputs = layers.Input(shape=(CFG['k']*9))
    km_inputs = layers.Input(shape=(CFG['n_clusters']))
    pca_inputs = layers.Input(shape=(CFG['n_components']))
    
    #----------- Embedding layers ----------------------
    embed = layers.Embedding(
        input_dim=emb_dims[0], 
        output_dim=emb_dims[1],
        embeddings_regularizer='l2'
        )(conv_inputs)

    #----------- Convolution1 layers ----------------------
    embed = layers.Conv1D(8, 1, activation='relu')(embed)
    embed = layers.Flatten()(embed)
    hidden1 = layers.Dropout(0.4)(embed)

    #----------- Convolution2 layers ----------------------
    cnv = layers.Conv2D(8, 3, padding='same', activation='relu')(conv2_inputs)
    cnv = layers.BatchNormalization()(cnv)
    cnv = layers.Conv2D(8, 3, padding='same', activation='relu')(cnv)
    cnv = layers.BatchNormalization()(cnv)
    cnv = layers.Flatten()(cnv)
    hidden2 = layers.Dropout(0.4)(cnv)

    #----------- Residual blocks layers ----------------------
    hidden1 = tfa.layers.NoisyDense(units=16, activation='relu')(hidden1)
    hidden1 = tfa.layers.WeightNormalization(
        layers.Dense(
            units=16,
            activation='relu',
            kernel_initializer='he_normal'
            ))(hidden1)
    hidden2 = tfa.layers.NoisyDense(units=16, activation='relu')(hidden2)
    hidden2 = tfa.layers.WeightNormalization(
        layers.Dense(
            units=16,
            activation='relu',
            kernel_initializer='he_normal'
            ))(hidden2)
    output = layers.Dropout(0.4)(layers.Concatenate()([embed, hidden1, hidden2, knn_inputs, km_inputs, pca_inputs]))
   
    output = tfa.layers.WeightNormalization(
        layers.Dense(
            units=64,
            activation='relu',
            kernel_initializer='he_normal'
        ))(output)
    output = layers.Dropout(0.2)(layers.Concatenate()([embed, hidden1, hidden2, knn_inputs, km_inputs, pca_inputs, output]))

    output = tfa.layers.WeightNormalization(
        layers.Dense(
            units=16,
            activation='relu',
            kernel_initializer='he_normal'
        ))(output)
   
    #----------- Final layer -----------------------
    conv_outputs = layers.Dense(
        units=9, 
        activation='softmax',
        kernel_initializer='lecun_normal')(output)
    
    #----------- Model instantiation  ---------------
    model = Model([conv_inputs, conv2_inputs, knn_inputs, km_inputs, pca_inputs], conv_outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tfa.optimizers.AdamW(learning_rate=CFG['learning_rate'], weight_decay=1e-7, amsgrad=True), 
        metrics=custom_metric
    )
    
    return model

create_model(emb_dims).summary()

oof = np.zeros((train.shape[0], CFG['n_class']))
pred = 0

skf = StratifiedKFold(n_splits=CFG['n_splits'], shuffle=True, random_state=CFG['seed'])

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target)):
    X_train, y_train = train[features].iloc[trn_idx], target_ohe.iloc[trn_idx]
    X_valid, y_valid = train[features].iloc[val_idx], target_ohe.iloc[val_idx]
    X_test = test[features]

    X_train_2d, X_valid_2d = train_2d[trn_idx], train_2d[val_idx]
    X_test_2d = test_2d

    X_train_knn, X_valid_knn = train_knn[trn_idx], train_knn[val_idx]
    X_test_knn = test_knn

    X_train_km, X_valid_km = train_km[trn_idx], train_km[val_idx]
    X_test_km = test_km

    X_train_pca, X_valid_pca = train_pca[trn_idx], train_pca[val_idx]
    X_test_pca = test_pca

    K.clear_session()  

    model = create_model(emb_dims)
    model.fit(
        [X_train, X_train_2d, X_train_knn, X_train_km, X_train_pca], y_train,
        batch_size=CFG['batch_size'],
        epochs=CFG['max_epochs'],
        validation_data=([X_valid, X_valid_2d, X_valid_knn, X_valid_km, X_valid_pca], y_valid),
        callbacks=[es, lr_scheduler],
        verbose=CFG['verbose']
    )

    oof[val_idx] = model.predict([X_valid, X_valid_2d, X_valid_knn, X_valid_km, X_valid_pca])
    pred += model.predict([X_test, X_test_2d, X_test_knn, X_test_km, X_test_pca]) / CFG['n_splits']
    m_logloss = log_loss(y_valid, oof[val_idx])
    print(f"fold{fold}: m_logloss {m_logloss}")
    
m_logloss = log_loss(target, oof)
print("-"*60)
print(f"m_logloss {m_logloss}")

np.save(SAVE_PATH + "cnn_oof", oof)
outdict = {}
outdict['id'] = test['id']
for i in range(9):
    outdict['Class_'+str(i+1)] = pred[:,i]
output = pd.DataFrame(outdict,index=None)
output.to_csv(SAVE_PATH+"cnn_pred.csv", index=False)