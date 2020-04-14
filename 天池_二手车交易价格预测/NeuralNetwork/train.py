import sys
sys.path.append('/home/aistudio/package')

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch_optimizer as newoptim

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import models

print('loading data')
train_path = './data/train_data_180.csv'
test_path = './data/test_data_180.csv'

train = pd.read_csv(train_path)
print('data loaded')

y = np.expm1(train['price'])
x = train.drop(['price', 'SaleID', 'regDate'], axis=1)
tags = x.columns
print(len(tags))
# test = pd.read_csv(test_path)

#归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x[tags].values)
x = min_max_scaler.transform(x[tags].values)

kfold = KFold(10, shuffle=True, random_state=233) 


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]
 
    def __getitem__(self, idx):
        batch_x = torch.from_numpy(np.array(self.X[idx]))
        batch_x = batch_x.float()
        batch_y = torch.from_numpy(np.array(self.y.iloc[idx]))
        batch_y = batch_y.float()
        
        return batch_x, batch_y

        
def train(model, epoch, train_loader, criterion):
    model.train()
    running_loss = 0.0
    total_size = 0
    for i, (input, target) in tqdm(enumerate(train_loader)):
        input = input.cuda(non_blocking=True)
        target = target.view(-1, 1)
        target = target.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        # compute output
        output = model(input)
        loss = criterion(output, target)
        running_loss += loss.data
        total_size += input.shape[0]
            
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        
    return running_loss / total_size
    
def validate(model, epoch, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    total_size = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.view(-1, 1)
            target = target.cuda(non_blocking=True)
            
            # compute output
            output = model(input)
            loss = criterion(output, target)
            running_loss += loss.data
            total_size += input.shape[0]
                
    return running_loss / total_size

train_batch_size = 2048
val_batch_size = 1024
epochs = 150
restart = epochs//3
lr = 1e-1
best_losses = {}

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(x)):
    if fold_idx < 7:
        continue
    if fold_idx > 7:
        break
    print('*******************fold {}*****************'.format(fold_idx))
    train_x = x[train_idx]
    train_y = y.iloc[train_idx]
    val_x = x[val_idx]
    val_y = y.iloc[val_idx]
    
    train_loader = torch.utils.data.DataLoader(MyDataset(train_x, train_y), batch_size=train_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(MyDataset(val_x, val_y), batch_size=val_batch_size, shuffle=False)
    model = models.MyResnet(len(tags))
    model.cuda()
    
    criterion = nn.L1Loss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    # optimizer = newoptim.RAdam(model.parameters(), lr=lr)
    
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=restart)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True, cooldown=1, factor=0.7, min_lr=1e-5)
    
    best_loss = 10000
    snapshot = 0
    count = 0
    
    for epoch in range(epochs):
        print('Epoch: ', epoch)
        train_loss = train(model, epoch, train_loader, criterion)
        val_loss = validate(model, epoch, val_loader, criterion)
        scheduler.step(val_loss)
        
        if val_loss <= best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), './checkpoint/fold{}_snap{}.pth'.format(fold_idx, snapshot))
            count = 0
        else:
            count += 1
        print('train_loss: {:4f}  val_loss: {:4f}  best_loss: {:4f}  lr: {}'.format(train_loss, val_loss, best_loss, optimizer.param_groups[0]['lr']))
        
        if count > 20:
            print('early stop at epoch {}'.format(epoch))
            count = 0
            break
    
    best_losses['fold{}'.format(fold_idx)] = best_loss.data.item()
    print(best_losses)
print(best_losses)

        
    
        
        
    
    
    
    
    