To run the wheremask.py use the below code
!python wheremask.py --original_path '/content/GeNeVA_datasets/data/iCLEVR/clevr_train.h5' --destination_path 'clevr_train_where_mask.h5'

To generate both WHERE and WHAT masks
give the path of the original h5 and destination h5

!python /content/generate_masks.py --o '/content/GeNeVA_datasets/data/iCLEVR/clevr_train.h5' --d 'clevr_train_with_mask.h5'
