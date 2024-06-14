## Step 1:

install avhubert following https://github.com/facebookresearch/av_hubert
x
then install transformers(4.32.0) and torch (2.0.0)

## Step2:

```cd av_hubert/fairseq/examples/```

then git clone this repo

## Step3: prepare avhubert features

```
cd msdm/data; python prepare_data.py --data-dir /data/to/MSDM_corpus --out-dir data/; cd ..
bash extract_feat.sh
```

Or can download the data and feats from xxx

## Step4: Train the classification model (conv3 and resnet)
```bash run.sh```

Or can download the checkpoints from xxx

## Step5: Produce predition
```bash predict.sh```
