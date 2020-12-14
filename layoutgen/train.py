#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import os.path
import random

# preprocessing imports
import joblib
import numpy as np

import math

# pytorch imports
import torch
import torch.utils.data
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint


# In[2]:


class ICLEVRLayoutDataset(Dataset):
    def __init__(self, cache_text_path="/users/jason wu/downloads/GeNeVA-v1/i-CLEVR/cache_text/", cache_scene_path="/users/jason wu/downloads/GeNeVA-v1/i-CLEVR/cache_scenes/", max_seq_len=5, split="train"):
        super(ICLEVRLayoutDataset, self).__init__()
        self.cache_text_path = cache_text_path
        self.cache_scene_path = cache_scene_path
        self.max_seq_len = max_seq_len
        self.split = split
        
        self.keys = [f.split(".")[0] for f in os.listdir(cache_text_path) if ".pkl" in f and split in f]
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        idx = idx % len(self.keys)
        key = self.keys[idx]
        key_file = key + ".pkl"
        txt_path = os.path.join(self.cache_text_path, key_file)
        scene_path = os.path.join(self.cache_scene_path, key_file)
        txt_data = joblib.load(txt_path)
        scene_data = joblib.load(scene_path)
        txt_data = torch.stack(txt_data, dim=0)
        orig_len = txt_data.shape[0]
        txt_data_padded = torch.zeros(self.max_seq_len, txt_data.shape[-1])
        txt_data_padded[:orig_len, :] = txt_data
        last_scene = torch.FloatTensor(scene_data[0])
        target_scene = torch.FloatTensor(scene_data[1])
        last_scene_padded = torch.zeros(self.max_seq_len, target_scene.shape[-1])
        if orig_len > 1:
            last_scene_padded[:last_scene.shape[0], :] = last_scene
        
        target_scene_padded = torch.zeros(self.max_seq_len, target_scene.shape[-1])
        target_scene_padded[:target_scene.shape[0], :] = target_scene
#         if last_scene.shape[0] == 0:
#             last_scene_padded = torch.zeros(target_scene.shape[-1])
#         else:
#             last_scene_padded = last_scene[last_scene.shape[0] - 1]
#         target_scene_padded = target_scene[target_scene.shape[0] - 1]
        element_dict = {}
        element_dict['txt'] = txt_data_padded
        element_dict['seq_len'] = orig_len
        element_dict['last_scene'] = last_scene_padded
        element_dict['target_scene'] = target_scene_padded
#         return txt_data_padded, last_scene_padded, target_scene_padded
        return element_dict


# In[3]:


ds = ICLEVRLayoutDataset()


# In[4]:


class ICLEVRLayoutDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=0, **kwargs):
        super(ICLEVRLayoutDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        # instantiate different splits of dataset
        self.train_dataset = ICLEVRLayoutDataset()
        self.dataloader_obj = DataLoader

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--num_workers", type=int, default=8)
        return parser
    
    def train_dataloader(self):
        return self.dataloader_obj(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


# In[5]:


data = ICLEVRLayoutDataModule(batch_size=4)


# In[6]:


dl = data.train_dataloader()


# In[7]:


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# In[8]:


class ICLEVRLayoutGen(pl.LightningModule):
    def __init__(self, num_heads=2, hidden_size=32, num_layers=2, dropout=0.25, lr=1e-4, num_shapes=3, num_colors=8, weight_decay=0, query_dim=100, scene_dim=13, **kwargs):
        super(ICLEVRLayoutGen, self).__init__()
        # handle argparse format of booleans
        self.save_hyperparameters()
        
        self.query_encoder = nn.Sequential(nn.Linear(query_dim, hidden_size), nn.LayerNorm(hidden_size))
        self.scene_encoder = nn.Sequential(nn.Linear(scene_dim, hidden_size), nn.LayerNorm(hidden_size))
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        transformer_layers = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.transformer = nn.TransformerEncoder(transformer_layers, num_layers)
        self.scene_decoder = nn.Linear(hidden_size, scene_dim)
    
    def forward(self, queries, scenes, paddingMask=None):
        query_hidden = self.query_encoder(queries)
        scene_hidden = self.scene_encoder(scenes)
        embedded = query_hidden + scene_hidden
        embedded *= np.sqrt(self.hparams.hidden_size)
        embedded = embedded.transpose(0, 1)
        embedded = self.pos_encoder(embedded)
        embedded = self.transformer(embedded, src_key_padding_mask=paddingMask)
        embedded = embedded.transpose(0, 1)
        return self.scene_decoder(embedded)
        
    
    def training_step(self, batch, batch_idx):
        batchSize = batch['txt'].shape[0]
        maxSeqLen = batch['txt'].shape[1]
        screenLen = batch['seq_len']
        
        sceneTxt = batch['txt']
        sceneLast = batch['last_scene']
        sceneTarget = batch['target_scene']
        
        sceneLast[:, :, 0] /= 320
        sceneLast[:, :, 1] /= 240
        sceneTarget[:, :, 0] /= 320
        sceneTarget[:, :, 1] /= 240
        
        _, targetShapeInd = sceneTarget[:, :, 10:].max(dim=-1)
        _, targetColorInd = sceneTarget[:, :, 2:10].max(dim=-1)
        
        # generate padding mask
        paddingMask = torch.arange(maxSeqLen, device=self.device).repeat(batchSize, 1) < screenLen.reshape(-1, 1).repeat(1, maxSeqLen)
        # invert padding mask - src_key_padding_mask uses True to mask
        paddingMask = ~paddingMask
        res = self.forward(sceneTxt, sceneLast, paddingMask=paddingMask)
        
        paddingMask = ~paddingMask
        preds = res[paddingMask]
        targetScene = sceneTarget[paddingMask]
        
        loss_pos = F.l1_loss(preds[:, :2], targetScene[:, :2])
        loss_shape = F.cross_entropy(preds[:, 10:].reshape(-1, 3), targetShapeInd[paddingMask].reshape(-1))
        loss_color = F.cross_entropy(preds[:, 2:2+8].reshape(-1, 8), targetColorInd[paddingMask].reshape(-1))
        
#         print("res", res[:, :, 5:].max(dim=-1)[1], "target", targetColorInd)
        
        loss = loss_pos + loss_shape + loss_color
        
        metrics = {'loss': loss, 'log': {'loss': loss, 'loss_pos': loss_pos, 'loss_shape': loss_shape, 'loss_color': loss_color}}
        return metrics
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, betas=(0.5, 0.999))
        return optimizer


# In[10]:


model = ICLEVRLayoutGen()

# In[11]:


trainer = pl.Trainer(max_epochs=10, callbacks=[ModelCheckpoint(filepath="{epoch:02d}")])

trainer.fit(model, data)

