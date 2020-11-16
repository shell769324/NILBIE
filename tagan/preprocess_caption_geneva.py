import os
import argparse
import fasttext as fastText

from data import ConvertCapVec

import torch.nn as nn

import pytorch_lightning as pl
from pl_bolts.models.autoencoders import AE

parser = argparse.ArgumentParser()
parser.add_argument('--caption_root', type=str, required=True,
                    help='root directory that contains captions')
parser.add_argument('--fasttext_model', type=str, required=True,
                    help='pretrained fastText model (binary file)')
parser.add_argument('--max_nwords', type=int, default=50,
                    help='maximum number of words (default: 50)')
parser.add_argument('--img_model', type=str, required=True, help='pretrained autoencoder model')
args = parser.parse_args()


if __name__ == '__main__':
    caption_root = args.caption_root.split('/')[-1]
    if (caption_root + '_vec') not in os.listdir(args.caption_root.replace(caption_root, '')):
        os.makedirs(args.caption_root + '_vec')
        print('Loading a pretrained image model...')
        img_model = AE.load_from_checkpoint(checkpoint_path=args.img_model)
        model = nn.Sequential(img_model.encoder, img_model.fc)
        model = model.eval()
        print('Loading a pretrained fastText model...')
        word_embedding = fastText.load_model(args.fasttext_model)
        print('Making vectorized caption data files...')
        ConvertCapVec().convert_and_save3(args.caption_root, word_embedding, args.max_nwords, model)
