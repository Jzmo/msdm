python train_dysarthria.py \
    --ckpt_path "checkpoints/base_vox_iter5.pt" \
    --train_data_dir "feats/base_vox_iter5/train/" \
    --eval_data_dir "feats/base_vox_iter5/valid/" \
    --output_dir "./exp/base_vox_iter5-resnet50/" \
    --classifier "resnet50" \
    --per_device_train_batch_size 64 \
    --logging_dir "./logs" \
    --logging_steps 500 \
    --evaluation_strategy "epoch" \
    --learning_rate 2e-5 \
    --num_train_epochs 30 \
    --num_classes 4

python train_dysarthria.py \
    --ckpt_path "checkpoints/base_vox_iter5.pt" \
    --train_data_dir "feats/base_vox_iter5/train/" \
    --eval_data_dir "feats/base_vox_iter5/valid/" \
    --output_dir "./exp/base_vox_iter5-conv3/" \
    --per_device_train_batch_size 64 \
    --classifier "conv3" \
    --evaluation_strategy "epoch" \
    --num_train_epochs 30 \
    --num_classes 4

