import sys

sys.path.append('/home/aistudio/package')

import os
import gc
import datetime
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Reshape, Flatten, Lambda
from keras.layers.merge import concatenate, dot, add, multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.regularizers import l1, l2, l1_l2
from keras.initializers import RandomUniform
from keras.optimizers import RMSprop, Adam, SGD
import keras

from nn_generator import DataGenerator

######################################################
## Data Loading
######################################################

folder = 'training'

## load train data
train = pd.read_csv('../music_test/train_nn.csv')
train_y = train['target']
train.drop(['target'], inplace=True, axis=1)

test = pd.read_csv('../music_test/test_nn.csv')
test_id = test['id']
test.drop(['id'], inplace=True, axis=1)


print('Train data loaded.')

## load other data
member = pd.read_csv('../music_test/members_nn.csv').sort_values('msno')
song = pd.read_csv('../music_test/songs_nn.csv').sort_values('song_id')

print('Member/Song data loaded.')

######################################################
## Feature Preparation
######################################################

## data preparation
## data preparation
train = train.merge(member[['msno', 'city', 'gender', 'registered_via', \
        'msno_rec_cnt', 'membership_days', 'registration_year', 'registration_month', 'registration_init_time', \
        'registration_date', 'expiration_year', 'expiration_month','bd_missing']], on='msno', how='left')
test = test.merge(member[['msno', 'city', 'gender', 'registered_via', \
         'msno_rec_cnt', 'membership_days', 'registration_year', 'registration_month', 'registration_init_time', \
        'registration_date', 'expiration_year', 'expiration_month','bd_missing']], on='msno', how='left')

train = train.merge(song[['song_id', 'artist_name', 'composer', 'lyricist', \
        'language', 'first_genre_id', 'second_genre_id', 'third_genre_id', \
        'cn', 'xxx','year','is_featured', 'isrc_missing', 'song_id_missing']], on='song_id', how='left')
test = test.merge(song[['song_id', 'artist_name', 'composer', 'lyricist', \
        'language', 'first_genre_id', 'second_genre_id', 'third_genre_id', \
        'cn', 'xxx','year','is_featured', 'isrc_missing', 'song_id_missing']], on='song_id', how='left')

gc.collect()

print('Data preparation done.')

## generate data for training
embedding_features = ['msno', 'city', 'gender', 'registered_via', 'bd_missing', \
        'artist_name', 'composer', 'lyricist', 'language', 'cn', 'xxx', 'is_featured', 'isrc_missing', 'song_id_missing', \
        'source_type', 'source_screen_name', 'source_system_tab', \
        'song_id']

train_embeddings = []
test_embeddings = []
for feat in embedding_features:
    train_embeddings.append(train[feat].values)
    test_embeddings.append(test[feat].values)

genre_features = ['first_genre_id', 'second_genre_id', 'third_genre_id']

train_genre = []
test_genre = []
for feat in genre_features:
    train_genre.append(train[feat].values)
    test_genre.append(test[feat].values)

context_features = [ 'msno_till_now_cnt', \
       'song_till_now_cnt', 'msno_10_before_cnt', 'song_10_before_cnt',\
       'msno_10_after_cnt', 'song_10_after_cnt', 'msno_25_before_cnt',\
       'song_25_before_cnt', 'msno_25_after_cnt', 'song_25_after_cnt',\
       'msno_500_before_cnt', 'song_500_before_cnt', 'msno_500_after_cnt',\
       'song_500_after_cnt', 'msno_5000_before_cnt', 'song_5000_before_cnt',\
       'msno_5000_after_cnt', 'song_5000_after_cnt', 'msno_10000_before_cnt',\
       'song_10000_before_cnt', 'msno_10000_after_cnt', 'song_10000_after_cnt',\
       'msno_50000_before_cnt', 'song_50000_before_cnt','msno_50000_after_cnt', \
       'song_50000_after_cnt', 'msno_source_system_tab_prob','msno_source_screen_name_prob',\
       'msno_source_type_prob','song_embeddings_dot', 'registration_init_time', 'timestamp']

train_context = train[context_features].values
test_context = test[context_features].values

ss_context = StandardScaler()
train_context = ss_context.fit_transform(train_context)
test_context = ss_context.transform(test_context)

del train
del test
gc.collect()


usr_features = ['bd', 'expiration_year', 'expiration_month', 'msno_rec_cnt', 
        'msno_source_screen_name_0', \
        'msno_source_screen_name_1', 'msno_source_screen_name_2', 'msno_source_screen_name_3', \
        'msno_source_screen_name_4', 'msno_source_screen_name_5', 'msno_source_screen_name_6', \
        'msno_source_screen_name_7', 'msno_source_screen_name_8', 'msno_source_screen_name_9', \
        'msno_source_screen_name_10', 'msno_source_screen_name_11', 'msno_source_screen_name_12', \
        'msno_source_screen_name_13', 'msno_source_screen_name_14', 'msno_source_screen_name_15', \
        'msno_source_screen_name_16', 'msno_source_screen_name_17', 'msno_source_screen_name_18', \
        'msno_source_screen_name_19', 'msno_source_screen_name_20', 'msno_source_screen_name_21', \
        'msno_source_system_tab_0', \
        'msno_source_system_tab_1', 'msno_source_system_tab_2', 'msno_source_system_tab_3', \
        'msno_source_system_tab_4', 'msno_source_system_tab_5', 'msno_source_system_tab_6', \
        'msno_source_system_tab_7', \
        'msno_source_type_0', 'msno_source_type_1', 'msno_source_type_10', \
        'msno_source_type_11', 'msno_source_type_2', \
        'msno_source_type_3', 'msno_source_type_4', 'msno_source_type_5', \
        'msno_source_type_6', \
        'msno_source_type_7', 'msno_source_type_8', 'msno_source_type_9', \
        'membership_days', 'registration_year','registration_month', 'registration_date']

usr_feat = member[usr_features].values
usr_feat = StandardScaler().fit_transform(usr_feat)

song_features = ['artist_rec_cnt', 'artist_song_cnt', 'composer_song_cnt', \
        'genre_rec_cnt', 'genre_song_cnt', 'song_length', \
        'song_rec_cnt', 'xxx_rec_cnt', 'xxx_song_cnt', 'year', 'year_song_cnt','year_rec_cnt',\
        'cn_song_cnt', 'cn_rec_cnt', 'composer_rec_cnt', 'lyricist_rec_cnt', 'genre_id_cnt', \
        'artist_cnt', 'lyricist_cnt', 'composer_cnt', 'is_featured', 'lyricist_song_cnt', 'cn', 'xxx',\
        'year', 'repeat_play_chance', 'plays']

song_feat = song[song_features].values
song_feat = StandardScaler().fit_transform(song_feat)

n_factors = 48

usr_component_features = ['member_component_%d'%i for i in range(n_factors)]
song_component_features = ['song_component_%d'%i for i in range(n_factors)]

usr_component = member[usr_component_features].values
song_component = song[song_component_features].values

n_artist = 16

usr_artist_features = ['member_artist_component_%d'%i for i in range(n_artist)]
song_artist_features = ['artist_component_%d'%i for i in range(n_artist)]

usr_artist_component = member[usr_artist_features].values
song_artist_component = song[song_artist_features].values

del member
del song
gc.collect()

dataGenerator = DataGenerator()

train_flow = dataGenerator.flow(train_embeddings+train_genre, [train_context], \
        train_y, batch_size=64000, shuffle=True)

######################################################
## Model Structure
######################################################

## define the model
def FunctionalDense(n, x, batchnorm=False, act='relu', lw1=0.0, dropout=None, name=''):
    if lw1 == 0.0:
        x = Dense(n, name=name+'_dense')(x)
    else:
        x = Dense(n, kernel_regularizer=l1(lw1), name=name+'_dense')(x)
    
    if batchnorm:
        x = BatchNormalization(name=name+'_batchnorm')(x)
        
    if act in {'relu', 'tanh', 'sigmoid'}:
        x = Activation(act, name=name+'_activation')(x)
    elif act =='prelu':
        x = PReLU(name=name+'_activation')(x)
    elif act == 'leakyrelu':
        x = LeakyReLU(name=name+'_activation')(x)
    elif act == 'elu':
        x = ELU(name=name+'_activation')(x)
    
    if dropout:
        if dropout > 0:
            x = Dropout(dropout, name=name+'_dropout')(x)
        
        
    return x

# def my_categorical_crossentropy(y_true, y_pred):
    # return keras.losses.binary_crossentropy(y_true, y_pred+1e-5)

def get_model(K, K0, lw=1e-4, lw1=1e-4, lr=1e-3, act='relu', batchnorm=False):
    embedding_inputs = []
    embedding_outputs = []
    for i in range(len(embedding_features) - 1):
        val_bound = 0.0 if i == 0 else 0.005
        tmp_input = Input(shape=(1,), dtype='int32', name=embedding_features[i]+'_input')
        max_value = int(train_embeddings[i].max()+1) if int(train_embeddings[i].max()+1) > int(test_embeddings[i].max()+1) else int(test_embeddings[i].max()+1)
        tmp_embeddings = Embedding(max_value,
                K if i == 0 else K0,
                embeddings_initializer=RandomUniform(minval=-val_bound, maxval=val_bound),
                embeddings_regularizer=l2(lw),
                input_length=1,
                trainable=True,
                name=embedding_features[i]+'_embeddings')(tmp_input)
        tmp_embeddings = Flatten(name=embedding_features[i]+'_flatten')(tmp_embeddings)
        
        embedding_inputs.append(tmp_input)
        embedding_outputs.append(tmp_embeddings)

    song_id_input = Input(shape=(1,), dtype='int32', name='song_id_input')

    embedding_inputs.append(song_id_input)
    
    genre_inputs = []
    genre_outputs = []
    max_value = int(np.max(train_genre)+1) if int(np.max(train_genre)+1) > int(np.max(test_genre)+1) else int(np.max(test_genre)+1)
    genre_embeddings = Embedding(max_value,
            K0,
            embeddings_initializer=RandomUniform(minval=-0.05, maxval=0.05),
            embeddings_regularizer=l2(lw),
            input_length=1,
            trainable=True,
            name='genre_embeddings')
    for i in range(len(genre_features)):
        tmp_input = Input(shape=(1,), dtype='int32', name=genre_features[i]+'_input')
        tmp_embeddings = genre_embeddings(tmp_input)
        tmp_embeddings = Flatten(name=genre_features[i]+'_flatten')(tmp_embeddings)
        
        genre_inputs.append(tmp_input)
        genre_outputs.append(tmp_embeddings)

    usr_input = Embedding(usr_feat.shape[0],
            usr_feat.shape[1],
            weights=[usr_feat],
            input_length=1,
            trainable=False,
            name='usr_feat')(embedding_inputs[0])
    usr_input = Flatten(name='usr_feat_flatten')(usr_input)
    
    song_input = Embedding(song_feat.shape[0],
            song_feat.shape[1],
            weights=[song_feat],
            input_length=1,
            trainable=False,
            name='song_feat')(song_id_input)
    song_input = Flatten(name='song_feat_flatten')(song_input)
    
    usr_component_input = Embedding(usr_component.shape[0],
            usr_component.shape[1],
            weights=[usr_component],
            input_length=1,
            trainable=False,
            name='usr_component')(embedding_inputs[0])
    usr_component_input = Flatten(name='usr_component_flatten')(usr_component_input)
    
    song_component_embeddings = Embedding(song_component.shape[0],
            song_component.shape[1],
            weights=[song_component],
            input_length=1,
            trainable=False,
            name='song_component')
    song_component_input = song_component_embeddings(song_id_input)
    song_component_input = Flatten(name='song_component_flatten')(song_component_input)

    usr_artist_component_input = Embedding(usr_artist_component.shape[0],
            usr_artist_component.shape[1],
            weights=[usr_artist_component],
            input_length=1,
            trainable=False,
            name='usr_artist_component')(embedding_inputs[0])
    usr_artist_component_input = Flatten(name='usr_artist_component_flatten')(usr_artist_component_input)
    
    song_artist_component_embeddings = Embedding(song_artist_component.shape[0],
            song_artist_component.shape[1],
            weights=[song_artist_component],
            input_length=1,
            trainable=False,
            name='song_artist_component')
    song_artist_component_input = song_artist_component_embeddings(song_id_input)
    song_artist_component_input = Flatten(name='song_artist_component_flatten')(song_artist_component_input)

    context_input = Input(shape=(len(context_features),), name='context_feat')
    
    # basic profiles
    usr_profile = concatenate(embedding_outputs[1:5]+[usr_input, \
            usr_component_input, usr_artist_component_input], name='usr_profile')
    song_profile = concatenate(embedding_outputs[5:14]+genre_outputs+[song_input, \
            song_component_input, song_artist_component_input], name='song_profile')

    multiply_component = dot([usr_component_input, song_component_input], axes=1, name='component_dot')
    multiply_artist_component = dot([usr_artist_component_input, \
            song_artist_component_input], axes=1, name='artist_component_dot')
    context_profile = concatenate(embedding_outputs[14:]+[context_input, multiply_component, multiply_artist_component], name='context_profile')
    
    
    # user field
    usr_embeddings = FunctionalDense(K*2, usr_profile, lw1=lw1, batchnorm=batchnorm, act=act, name='usr_profile')
    usr_embeddings = Dense(K, name='usr_profile_output')(usr_embeddings)
    usr_embeddings = add([usr_embeddings, embedding_outputs[0]], name='usr_embeddings')
    
    # song field
    song_embeddings = FunctionalDense(K*2, song_profile, lw1=lw1, batchnorm=batchnorm, act=act, name='song_profile')
    song_embeddings = Dense(K, name='song_profile_output')(song_embeddings)
    #song_embeddings = add([song_embeddings, embedding_outputs[4]], name='song_embeddings')
    
    # context field
    context_embeddings = FunctionalDense(K, context_profile, lw1=lw1, batchnorm=batchnorm, act=act, name='context_profile')

    # joint embeddings
    joint = dot([usr_embeddings, song_embeddings], axes=1, normalize=False, name='pred_cross')
    joint_embeddings = concatenate([usr_embeddings, song_embeddings, context_embeddings, joint], name='joint_embeddings')
    
    # top model
    preds0 = FunctionalDense(K*2, joint_embeddings, batchnorm=batchnorm, act=act, name='preds_0')
    preds1 = FunctionalDense(K*2, concatenate([joint_embeddings, preds0]), batchnorm=batchnorm, act=act, name='preds_1')
    preds2 = FunctionalDense(K*2, concatenate([joint_embeddings, preds0, preds1]), batchnorm=batchnorm, act=act, name='preds_2')
    
    preds = concatenate([joint_embeddings, preds0, preds1, preds2], name='prediction_aggr')
    preds = Dropout(0.5, name='prediction_dropout')(preds)
    preds = Dense(1, activation='sigmoid', name='prediction')(preds)
        
    temp = embedding_inputs + genre_inputs + [context_input]
    model = Model(inputs=embedding_inputs + genre_inputs + [context_input], outputs=preds)
    
    opt = RMSprop(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

    return model

######################################################
## Model Training
######################################################

## train the model
para = pd.read_csv('./nn_record_header.csv').sort_values(by='val_auc', ascending=False)
for i in range(10, 18):
    K = para['K'].values[i]
    K0 = para['K0'].values[i]
    lw = para['lw'].values[i]
    lw1 = para['lw1'].values[i]
    lr = para['lr'].values[i]
    lr_decay = para['lr_decay'].values[i]
    activation = para['activation'].values[i]
    batchnorm = para['batchnorm'].values[i]
    bst_epoch = para['bst_epoch'].values[i]
    train_loss0 = para['trn_loss'].values[i]
    val_auc = para['val_auc'].values[i]
    sample_weight_rate = 0.0
    
    print('K: %d, K0: %d, lw: %e, lw1: %e, lr: %e, lr_decay: %f, act: %s, batchnorm: %s'%(K, K0, lw, \
            lw1, lr, lr_decay, activation, batchnorm))
    
    model = get_model(K, K0, lw, lw1, lr, activation, batchnorm)
 
    early_stopping =EarlyStopping(monitor='loss', patience=5, min_delta=0.0000)
    model_path = './train_checkpoint/bst_model_%s_%d.h5'%(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), np.random.randint(0,65536))
    model_checkpoint = ModelCheckpoint(model_path, monitor='loss', save_best_only=True, \
        save_weights_only=True)
    lr_reducer = LearningRateScheduler(lambda x: lr*(lr_decay**x))
    
    hist = model.fit_generator(train_flow, train_flow.__len__(), epochs=bst_epoch, workers=4, callbacks=[early_stopping, model_checkpoint, lr_reducer])
    
    train_loss = hist.history['loss'][-1]

    val_auc = para['val_auc'].values[i]
    print('Model training done. Validation AUC: %.5f'%val_auc)

    flag = np.random.randint(0, 65536)
    
    test_flow = dataGenerator.flow(test_embeddings + test_genre, [test_context], \
            batch_size=16384, shuffle=False)
    test_pred = model.predict_generator(test_flow, test_flow.__len__(), workers=1)
    
    test_sub = pd.DataFrame({'id': test_id, 'target': test_pred.ravel()})
    test_sub.to_csv('./train_checkpoint/nn_%.5f_%.5f_%d.csv'%(val_auc, train_loss, flag), index=False)
    
