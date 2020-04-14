import sys
sys.path.append('/home/aistudio/package')

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

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
train.fillna(0, inplace=True)
y = train['price']
train_x = train.drop(['price', 'SaleID', 'regDate'], axis=1)

test = pd.read_csv(test_path)
test.fillna(0, inplace=True)
id = test['SaleID']
data = test.drop(['SaleID', 'regDate'], axis=1)
tags = data.columns
print('data loaded')
print(len(data.columns))

#归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(train_x[tags].values)
x = min_max_scaler.transform(data[tags].values)

class MyDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]
 
    def __getitem__(self, idx):
        batch_x = torch.from_numpy(np.array(self.X[idx]))
        batch_x = batch_x.float()
        
        return batch_x   

def predict(model):
    res = []
    count = 0
    for saleid in tqdm(np.array(id)):
        inputx = torch.from_numpy(x[count])
        inputx = inputx.float()
        inputx = inputx.unsqueeze(0)
        output = model(inputx.cuda())
        res.append((saleid, output.data.item()))
        count += 1
    return res

batch_size=4096    
def predict_batch(model):
    res = []
    val_loader = torch.utils.data.DataLoader(MyDataset(x), batch_size=batch_size, shuffle=False)
    count = 150000
    for i, input_data in tqdm(enumerate(val_loader)):
        input_data = input_data.cuda()
        out = model(input_data)
        for data in out.cpu().data.numpy():
            res.append((count, data[0]))
            count += 1
    return res

# model = torch.load('./checkpoint/fold{}_snap0.pkl'.format(1))
# print(model)

model_numbers = 10
sub = []
for i in range(model_numbers):
    if i == 9:
        continue
    model = models.buildModel(models.MyResnet(len(tags)), model_path='./checkpoint/fold{}_snap0.pth'.format(i))
    model.eval()
    model.cuda()
    df = pd.DataFrame(data=predict_batch(model), columns=['SaleID', 'price'])
    sub.append(df)
    print('fold{}'.format(i), len(sub))


df = sub[0]
for i in range(1, len(sub)):
    temp = sub[i]
    df['price'] += temp['price']
df['price'] /= len(sub)
df.to_csv('./submit/submit_batch_0411_2.csv', index=False)
print('save')
    
    