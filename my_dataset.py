#mydataset
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
from torchvision import transforms as t
import pandas as pd

as_tensor = t.ToTensor()

class APdataset(Dataset):
    def __init__(self,img_roots:list, type=0,transforms=None, random_seed=233):
        self.datalist = r'./dataset.csv'
        self.tfms=transforms
        self.img_roots = img_roots
        
        data_DF = pd.read_csv(self.datalist)
        data_DF = data_DF.sample(frac=1,random_state=random_seed).reset_index(drop=True)
        
        if type==0:
            self.target=data_DF.loc[:2340].reset_index(drop=True)
            self.bias = 0
        if type==1:
            self.target=data_DF.loc[2340:2632].reset_index(drop=True)
            self.bias = 2340
        if type==2:
            self.target=data_DF.loc[2632:2924].reset_index(drop=True)
            self.bias = 2632
    def __getitem__(self, index):
        img_tensor = torch.empty((0,224,224))
        for dir in self.img_roots:
            img=Image.open(dir+'/'+self.target.loc[index][1]).convert('L')
            item = as_tensor(self.tfms(image=np.array(img))['image']) # [1, 224, 224]
            img_tensor = torch.concat((img_tensor,item),0)
        
        img_tensor = img_tensor.flatten(0,1).unsqueeze(0) # (8,224,224)->(1,8*224,224)
        target=torch.tensor(self.target.loc[index][2:].values.astype(int), dtype=torch.long)
        
        return img_tensor,target
    def __len__(self):
        return len(self.target)


