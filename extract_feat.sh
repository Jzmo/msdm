#!/bin/bash

python extract_avhubert_feats.py \
    --ckpt_path "checkpoints/base_vox_iter5.pt" \
    --manifest_path "data/test.tsv" \
    --sample_rate 16000 \
    --modalities "video" "audio" \
    --normalize \
    --output_dir "feats/base_vox_iter5/test"

exit 0
python extract_avhubert_feats.py \
    --ckpt_path "checkpoints/base_vox_iter5.pt" \
    --manifest_path "data/train.tsv" \
    --label_path "data/train.wrd" \
    --sample_rate 16000 \
    --modalities "video" "audio" \
    --normalize \
    --output_dir "feats/base_vox_iter5/train"

python extract_avhubert_feats.py \
    --ckpt_path "checkpoints/base_vox_iter5.pt" \
    --manifest_path "data/valid.tsv" \
    --label_path "data/valid.wrd" \
    --sample_rate 16000 \
    --modalities "video" "audio" \
    --normalize \
    --output_dir "feats/base_vox_iter5/valid"

