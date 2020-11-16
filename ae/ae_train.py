import argparse

import pytorch_lightning as pl

from pl_bolts.models.autoencoders import AE

from ae_datasets import iCLEVRDataModule
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser()

parser = AE.add_model_specific_args(parser)
parser = iCLEVRDataModule.add_dataset_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)

opt = parser.parse_args()

model = AE(**vars(opt))
data = iCLEVRDataModule(**vars(opt))

trainer = pl.Trainer.from_argparse_args(opt, log_gpu_memory='all', checkpoint_callback=ModelCheckpoint(filepath="{epoch:02d}.ckpt"))
trainer.fit(model, data)