#!/bin/bash
python predict_dysarthria.py --ckpt_path exp/base_vox_iter5-conv3/checkpoint-25980/pytorch_model.bin \
       --test_data_dir "feats/base_vox_iter5/test/" \
       --classifier conv3 \
       --num_classes 4 \
       --output_file exp/base_vox_iter5-conv3/checkpoint-25980/prediction.txt \


python predict_dysarthria.py --ckpt_path exp/base_vox_iter5-resnet50/checkpoint-25980/pytorch_model.bin \
       --test_data_dir "feats/base_vox_iter5/test/" \
       --classifier resnet50 \
       --num_classes 4 \
       --output_file exp/base_vox_iter5-resnet50/checkpoint-25980/prediction.txt \
