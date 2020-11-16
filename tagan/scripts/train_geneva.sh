. ./CONFIG

python train.py --img_root /storage/GeNeVA-v1/i-CLEVR/images/ --caption_root /storage/GeNeVA-v1/i-CLEVR/ --trainclasses_file /storage/GeNeVA-v1/i-CLEVR/trainvalclasses.txt --save_filename_G ./models/geneva_G.pth --save_filename_D ./models/geneva_D.pth --lambda_cond_loss 10 --lambda_recon_loss 0.2
