import sys

sys.path.append('/home/aistudio/package')

import os
import gc
import datetime
import numpy as np
import pandas as pd

# auc评价指标
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# keras相关
from keras.models import Model
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Reshape, Flatten, Lambda
from keras.layers.merge import concatenate, dot, add, multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.regularizers import l1, l2, l1_l2
from keras.initializers import RandomUniform
from keras.optimizers import RMSprop, Adam, SGD

from nn_generator import DataGenerator

## 1. 数据加载

# 加载训练集和验证集数据
train = pd.read_csv('../music_val/train_val_nn.csv')
test = pd.read_csv('../music_val/test_val_nn.csv')
member = pd.read_csv('../music_val/members_val_nn.csv').sort_values('msno')
song = pd.read_csv('../music_val/songs_val_nn.csv').sort_values('song_id')

train_y = train['target']
train.drop(['target'], inplace=True, axis=1)

test_y = test['target']
test.drop(['target'], inplace=True, axis=1)

print('Data loaded.')

## 2. 数据的准备

# 合并用到的用户信息
train = train.merge(member[['msno', 'city', 'gender', 'registered_via', \
        'msno_rec_cnt', 'membership_days', 'registration_year', 'registration_month', 'registration_init_time', \
        'registration_date', 'expiration_year', 'expiration_month','bd_missing']], on='msno', how='left')
test = test.merge(member[['msno', 'city', 'gender', 'registered_via', \
         'msno_rec_cnt', 'membership_days', 'registration_year', 'registration_month', 'registration_init_time', \
        'registration_date', 'expiration_year', 'expiration_month','bd_missing']], on='msno', how='left')

# 合并用到的歌曲信息
train = train.merge(song[['song_id', 'artist_name', 'composer', 'lyricist', \
        'language', 'first_genre_id', 'second_genre_id', 'third_genre_id', \
        'cn', 'xxx','year','is_featured', 'isrc_missing', 'song_id_missing']], on='song_id', how='left')
test = test.merge(song[['song_id', 'artist_name', 'composer', 'lyricist', \
        'language', 'first_genre_id', 'second_genre_id', 'third_genre_id', \
        'cn', 'xxx','year','is_featured', 'isrc_missing', 'song_id_missing']], on='song_id', how='left')

gc.collect()

print('Data preparation done.')

## 3. 特征分类

embedding_features = ['msno', 'city', 'gender', 'registered_via', \
        'artist_name', 'composer', 'lyricist', 'language', 'cn', 'xxx',\
        'source_type', 'source_screen_name', 'source_system_tab', 'song_id']

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
    
context_features = [ 'msno_till_now_cnt', 'song_till_now_cnt', 'msno_10_before_cnt', 'song_10_before_cnt',\
       'msno_10_after_cnt', 'song_10_after_cnt', 'msno_25_before_cnt', 'song_25_before_cnt', 'msno_25_after_cnt', \
       'song_25_after_cnt', 'msno_500_before_cnt', 'song_500_before_cnt', 'msno_500_after_cnt', 'song_500_after_cnt', 
       'msno_5000_before_cnt', 'song_5000_before_cnt', 'msno_5000_after_cnt', 'song_5000_after_cnt', \
       'msno_10000_before_cnt', 'song_10000_before_cnt', 'msno_10000_after_cnt', 'song_10000_after_cnt',\
       'msno_50000_before_cnt', 'song_50000_before_cnt','msno_50000_after_cnt', 'song_50000_after_cnt', \
       'msno_source_system_tab_prob','msno_source_screen_name_prob',\
       'msno_source_type_prob','song_embeddings_dot', 'artist_embeddings_dot',\
       'registration_init_time', 'timestamp',\
       'msno_artist_name_prob', 'msno_first_genre_id_prob', 'msno_xxx_prob','msno_language_prob', 'msno_year_prob', \
       'song_source_system_tab_prob','song_source_screen_name_prob', 'song_source_type_prob', 'msno_source_prob']

train_context = train[context_features].values
test_context = test[context_features].values

ss_context = StandardScaler()
train_context = ss_context.fit_transform(train_context)
test_context = ss_context.transform(test_context)

del train
del test
gc.collect()

usr_features = ['bd','msno_rec_cnt', 'msno_source_screen_name_0', 'msno_source_screen_name_1', 'msno_source_screen_name_10',\
        'msno_source_screen_name_11', 'msno_source_screen_name_12', 'msno_source_screen_name_13', 'msno_source_screen_name_14', \
        'msno_source_screen_name_17',  'msno_source_screen_name_18', 'msno_source_screen_name_19', 'msno_source_screen_name_2', \
        'msno_source_screen_name_3', 'msno_source_screen_name_4', 'msno_source_screen_name_5', 'msno_source_screen_name_6', \
        'msno_source_screen_name_7', 'msno_source_screen_name_8', 'msno_source_screen_name_9', 'msno_source_system_tab_0', \
        'msno_source_system_tab_1', 'msno_source_system_tab_2', 'msno_source_system_tab_3', 'msno_source_system_tab_4',\
        'msno_source_system_tab_5', 'msno_source_system_tab_6', 'msno_source_system_tab_7', 'msno_source_type_0',\
        'msno_source_type_1','msno_source_type_2','msno_source_type_3', 'msno_source_type_4', 'msno_source_type_5', \
        'msno_source_type_6', 'msno_source_type_7', 'msno_source_type_8', 'msno_source_type_9', 'msno_source_type_10', \
        'msno_source_type_11', 'membership_days', 'expiration_date','expiration_year', 'expiration_month', 'expiration_day',\
        'msno_timestamp_std', 'msno_song_length_mean', 'msno_artist_song_cnt_mean', 'msno_artist_rec_cnt_mean',\
        'msno_song_rec_cnt_mean', 'msno_year_mean', 'msno_song_length_std',\
        'msno_artist_song_cnt_std', 'msno_artist_rec_cnt_std','msno_song_rec_cnt_std', 'msno_year_std']

usr_feat = member[usr_features].values
usr_feat = StandardScaler().fit_transform(usr_feat)

song_features = ['song_length','genre_id_cnt','artist_rec_cnt', 'artist_song_cnt', 'composer_song_cnt', 'genre_rec_cnt', 'genre_song_cnt', \
        'song_rec_cnt', 'xxx_rec_cnt', 'xxx_song_cnt', 'year', 'year_song_cnt','year_rec_cnt', 'cn_song_cnt', 'cn_rec_cnt', 'composer_rec_cnt', \
        'lyricist_rec_cnt', 'genre_id_cnt', 'artist_cnt', 'lyricist_cnt', 'composer_cnt', 'lyricist_song_cnt', \
        'year', 'repeat_play_chance', 'plays','song_timestamp_mean', 'song_timestamp_std']

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
test_flow = dataGenerator.flow(test_embeddings+test_genre, [test_context], \
        test_y, batch_size=16384, shuffle=False)

## 4. 构建模型

# 定义全连接的参数配置
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

def return_input(x):
    return x
    

## 定义模型
# K是给msno用的，K0是给其他embedding特征用的
def get_model(K, K0, lw=1e-4, lw1=1e-4, lr=1e-3, act='relu', batchnorm=False):
    embedding_inputs = []
    embedding_outputs = []
    for i in range(len(embedding_features) - 1): # 去掉song_id
        val_bound = 0.0 if i == 0 else 0.005 # i==0时是用户id，即msno
        tmp_input = Input(shape=(1,), dtype='int32', name=embedding_features[i]+'_input')
        
        # max_value代表原来有多少种取值
        # K/K0代表想要变成多少种取值
        # tmp_embeddings得到的是n*1*K
        max_value = int(train_embeddings[i].max()+1) if int(train_embeddings[i].max()+1) > int(test_embeddings[i].max()+1) else int(test_embeddings[i].max()+1)
        tmp_embeddings = Embedding(max_value,
                K if i == 0 else K0,
                embeddings_initializer=RandomUniform(minval=-val_bound, maxval=val_bound),
                embeddings_regularizer=l2(lw),
                input_length=1,
                trainable=True,
                name=embedding_features[i]+'_embeddings')(tmp_input)
        # 把1*K拉平，就变成了K个数
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

    # weights中间矩阵，最后user_input得到的是这个用户在usr_feat中的所有特征，embedding_inputs[0]是具体的一个用户
    usr_input = Embedding(usr_feat.shape[0],
            usr_feat.shape[1],
            weights=[usr_feat],
            input_length=1,
            trainable=False,
            name='usr_feat')(embedding_inputs[0])
    usr_input = Flatten(name='usr_feat_flatten')(usr_input)
    
    # song_input得到的是这个歌曲id在song_feat中的所有特征，song_id_input是具体的一首歌
    song_input = Embedding(song_feat.shape[0],
            song_feat.shape[1],
            weights=[song_feat],
            input_length=1,
            trainable=False,
            name='song_feat')(song_id_input)
    song_input = Flatten(name='song_feat_flatten')(song_input)
    
    # usr_component_input 得到的是这个用户在usr_component中的所有特征
    usr_component_input = Embedding(usr_component.shape[0],
            usr_component.shape[1],
            weights=[usr_component],
            input_length=1,
            trainable=False,
            name='usr_component')(embedding_inputs[0])
    usr_component_input = Flatten(name='usr_component_flatten')(usr_component_input)
    
    # song_component_embeddings 得到的是这首歌曲在song_component中的所有特征
    song_component_embeddings = Embedding(song_component.shape[0],
            song_component.shape[1],
            weights=[song_component],
            input_length=1,
            trainable=False,
            name='song_component')
    song_component_input = song_component_embeddings(song_id_input)
    song_component_input = Flatten(name='song_component_flatten')(song_component_input)

    # usr_artist_component_input 得到的是这首歌曲在 usr_artist_component 中的所有特征
    usr_artist_component_input = Embedding(usr_artist_component.shape[0],
            usr_artist_component.shape[1],
            weights=[usr_artist_component],
            input_length=1,
            trainable=False,
            name='usr_artist_component')(embedding_inputs[0])
    usr_artist_component_input = Flatten(name='usr_artist_component_flatten')(usr_artist_component_input)
    
    # song_artist_component_embeddings 得到的是这首歌曲在 song_artist_component 中的所有特征
    song_artist_component_embeddings = Embedding(song_artist_component.shape[0],
            song_artist_component.shape[1],
            weights=[song_artist_component],
            input_length=1,
            trainable=False,
            name='song_artist_component')
    song_artist_component_input = song_artist_component_embeddings(song_id_input)
    song_artist_component_input = Flatten(name='song_artist_component_flatten')(song_artist_component_input)

    context_input = Input(shape=(len(context_features), ), name='context_feat')
    
    # 连接所有用户信息
    usr_profile = concatenate(embedding_outputs[1:5]+[usr_input, \
            usr_component_input, usr_artist_component_input], name='usr_profile')
    # 连接所有歌曲信息
    song_profile = concatenate(embedding_outputs[5:14]+genre_outputs+[song_input, \
            song_component_input, song_artist_component_input], name='song_profile')

    # 对SVD后的用户信息，歌曲信息做点积
    multiply_component = dot([usr_component_input, song_component_input], axes=1, name='component_dot')
    # 对SVD后的用户信息，歌手信息做点积
    multiply_artist_component = dot([usr_artist_component_input, song_artist_component_input], axes=1, name='artist_component_dot')
    # 得到最后的context信息
    context_profile = concatenate(embedding_outputs[14:]+[context_input, multiply_component, multiply_artist_component], name='context_profile')
    
    
    # 对用户信息做两层全连接，最后结果相加
    usr_embeddings = FunctionalDense(K*2, usr_profile, lw1=lw1, batchnorm=batchnorm, act=act, name='usr_profile')
    usr_embeddings = Dense(K, name='usr_profile_output')(usr_embeddings)
    usr_embeddings = add([usr_embeddings, embedding_outputs[0]], name='usr_embeddings')
    
    # 对歌曲信息做两层全连接
    song_embeddings = FunctionalDense(K*2, song_profile, lw1=lw1, batchnorm=batchnorm, act=act, name='song_profile')
    song_embeddings = Dense(K, name='song_profile_output')(song_embeddings)
    
    # 对context信息做一层全连接
    context_embeddings = FunctionalDense(K, context_profile, lw1=lw1, batchnorm=batchnorm, act=act, name='context_profile')

    # 对全连接后产生的user和song做点积，并对用户、歌曲、产生的点积结果、context四者做拼接
    joint = dot([usr_embeddings, song_embeddings], axes=1, normalize=False, name='pred_cross')
    joint_embeddings = concatenate([usr_embeddings, song_embeddings, context_embeddings, joint], name='joint_embeddings')
    
    # 上层的模型
    # 借鉴denseNet的网络结构，构建了一个三层的网络
    preds0 = FunctionalDense(K*2, joint_embeddings, batchnorm=batchnorm, act=act, name='preds_0')
    preds1 = FunctionalDense(K*2, concatenate([joint_embeddings, preds0]), batchnorm=batchnorm, act=act, name='preds_1')
    preds2 = FunctionalDense(K*2, concatenate([joint_embeddings, preds0, preds1]), batchnorm=batchnorm, act=act, name='preds_2')
    
    preds = concatenate([joint_embeddings, preds0, preds1, preds2], name='prediction_aggr')
    preds = Dropout(0.5, name='prediction_dropout')(preds)
    preds = Dense(1, activation='sigmoid', name='prediction')(preds)
    
    temp = embedding_inputs + genre_inputs + [context_input]
    model = Model(inputs=temp, outputs=preds)
    
    opt = RMSprop(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

    return model

## 5. 模型训练

# para = pd.read_csv('./nn_record.csv').sort_values(by='val_auc', ascending=False)
# 一共100轮，每轮随机初始化所有的参数，包括embedding后的维数，正则参数，学习率和学习率衰减系数，激活函数，batch归一化参数，样本权重，
for i in range(100):
    K = np.random.randint(48, 128)  # 64
    K0 = np.random.randint(4, 16)  # 8
    lw = 7.5e-4 * (0.1 ** (np.random.rand() * 3 - 1.5))
    lw1 = 0.0
    lr = 1e-2
    lr_decay = 0.65 + np.random.rand() * 0.3
    activation = np.random.choice(['relu', 'tanh', 'prelu', 'leakyrelu', 'elu'])
    batchnorm = np.random.choice([True, False])
    sample_weight_rate = 0.0
    
    print('K: %d, K0: %d, lw: %e, lw1: %e, lr: %e, lr_decay: %f, act: %s, batchnorm: %s'%(K, K0, lw, \
            lw1, lr, lr_decay, activation, batchnorm))
    
    model = get_model(K, K0, lw, lw1, lr, activation, batchnorm)
    
    early_stopping =EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0000)
    model_path = './checkpoint/bst_model_%s_%d.h5'%(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), np.random.randint(0,65536))
    # 保存最好结果的权重
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, \
            save_weights_only=True)
    lr_reducer = LearningRateScheduler(lambda x: lr*(lr_decay**x))
    
    # 生成器函数为.fit_generator函数生成一批大小为batch_size的数据
    # .fit_generator函数接受批量数据，执行反向传播，并更新模型中的权重，重复该过程直到达到期望的epoch数量
    # 每次训练跑40轮，验证集上一轮1-2分钟，测试集上3-5分钟。跑完验证集差不多1小时，测试集两三个小时
    hist = model.fit_generator(train_flow, train_flow.__len__(), epochs=40, \
            validation_data=test_flow, validation_steps=test_flow.__len__(), \
            workers=4, callbacks=[early_stopping, model_checkpoint, lr_reducer])

    # 把结果最好的参数加载进来
    model.load_weights(model_path)
    os.remove(model_path)

    # 输出验证集上auc最高，loss最低的分值
    bst_epoch = np.argmin(hist.history['val_loss'])
    trn_loss = hist.history['loss'][bst_epoch]
    trn_acc = hist.history['acc'][bst_epoch]
    val_loss = hist.history['val_loss'][bst_epoch]
    val_acc = hist.history['val_acc'][bst_epoch]

    # 生成器生成一批大小为batch_size的数据
    test_flow_predict = dataGenerator.flow(test_embeddings+test_genre, [test_context], batch_size=16000, shuffle=False)
    preds_val = model.predict_generator(test_flow_predict, test_flow_predict.__len__(), workers=1)
    val_auc = roc_auc_score(test_y, preds_val)
    print('Validation AUC: %.5f'%val_auc)
    
    # 保存本此迭代中结果最好的参数
    res = '%s,%s,%s,%s,%d,%d,%e,%e,%e,%f,%e,%d,%.5f,%.5f,%.5f,%.5f,%.5f\n'%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), \
            'nn_generator_one', activation, batchnorm, K, K0, lw, lw1, lr, lr_decay, sample_weight_rate, bst_epoch+1, trn_loss, \
            trn_acc, val_loss, val_acc, val_auc)
    f = open('./nn_record.csv', 'a')
    f.write(res)
    f.close()
    
