# system imports
from argparse import ArgumentParser
import os
import os.path

from PIL import Image

# pytorch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# pytorch lightning imports
import pytorch_lightning as pl

class iCLEVRDataset(Dataset):
    def __init__(self, data_path="/storage/GeNeVA-v1/i-CLEVR/images", input_height=256, mode="train", verbose=False, **kwargs):
        super(iCLEVRDataset, self).__init__()
        # save dataset parameters
        self.data_path = data_path
        self.input_height = input_height
        self.verbose = verbose
        
        img_transforms = transforms.Compose([
            transforms.Resize((input_height, input_height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.img_transforms = img_transforms
        
        # get keys in split
        keys = os.listdir(data_path)
        keys = [k for k in keys if len(k) > 0 and k[0] != "." and k.endswith(".png") and mode in k]
        keys = [os.path.join(data_path, k) for k in keys]
        self.keys = keys
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        idx = idx % len(self.keys)
        
        def tryAnother():
            return self.__getitem__(idx + 1)
        
#         try:
        imgPath = self.keys[idx]
        img = Image.open(imgPath).convert("RGB")
        img = self.img_transforms(img)
        return img, 0

#         except:
#             if self.verbose:
#                 print("failed", str(idx))
#             return tryAnother()
        
class iCLEVRDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_path="/storage/GeNeVA-v1/i-CLEVR/images", input_height=256, verbose=False, **kwargs):
        super(iCLEVRDataModule, self).__init__()
        if isinstance(verbose, int):
            verbose = verbose > 0
        self.batch_size = batch_size
        self.train_dataset = iCLEVRDataset(data_path=data_path, input_height=input_height, mode="train", verbose=verbose)
        self.val_dataset = iCLEVRDataset(data_path=data_path, input_height=input_height, mode="val", verbose=verbose)
        self.test_dataset = iCLEVRDataset(data_path=data_path, input_height=input_height, mode="test", verbose=verbose)
        
    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_path", type=str, default="/storage/GeNeVA-v1/i-CLEVR/images")
        parser.add_argument("--input_height", type=int, default=256)
        parser.add_argument("--verbose", type=int, default=0)
        return parser
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)