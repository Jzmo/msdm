import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys
import hubert_pretraining, hubert
from torch.utils.data import DataLoader
# import hubert
import tqdm
from hubert_classification_dataset import AVHubertFeatureDataset_FromFile
from dysarthria_model import Conv3AudioClassifier, ResNet50AudioClassifier
import fairseq
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
import argparse

"""
class DysClassifierTrainer(Trainer):
    def compute_loss(self, model, inputs,return_outputs=False):
        # Extract inputs and labels from the dictionary
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        return (loss, outputs) if return_outputs else loss

"""
def compute_model_size(model):
    num_params = sum(p.numel() for p in model.parameters())
    return num_params

def collate_fn(batch):
    features, labels, fids = zip(*batch)
    
    # Pad sequences to the same length
    padded_features = pad_sequence(features, batch_first=True)
    
    labels = torch.tensor(labels, dtype=torch.long)
    return {"features":padded_features, "labels":labels, "fids":fids}

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    parser = argparse.ArgumentParser(description="Train a ResNet50 model for audio classification")

    # Add arguments
    parser.add_argument('--ckpt_path', type=str, default="checkpoints/base_vox_iter5.pt", help='Path to the checkpoint file')
    parser.add_argument('--train_data_dir', type=str, default='feats/base_vox_iter5/train/', help='Directory containing training data')
    parser.add_argument('--eval_data_dir', type=str, default='feats/base_vox_iter5/valid/', help='Directory containing evaluation data')
    parser.add_argument('--output_dir', type=str, default='./exp/base_vox_iter5-conv3/', help='Directory to save the output')
    parser.add_argument('--per_device_train_batch_size', type=int, default=64, help='Batch size per device for training')
    parser.add_argument('--logging_dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--logging_steps', type=int, default=500, help='Number of steps between logging')
    parser.add_argument('--evaluation_strategy', type=str, default="epoch", help='Evaluation strategy')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_train_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes for the classifier')
    parser.add_argument('--classifier', type=str, default='conv3', choices=['resnet50', 'conv3'], help='Choose the classifier architecture')
    parser.add_argument('--save_strategy', type=str, default='epoch', help='Save strategy')
    parser.add_argument('--save_total_limit', type=int, default=3, help='Total limit of saved checkpoints')

    
    # Parse arguments
    args = parser.parse_args()

    # Check if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        evaluation_strategy=args.evaluation_strategy,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        output_dir=args.output_dir,
    )
    if args.classifier == 'resnet50':
        model = ResNet50AudioClassifier(num_classes=args.num_classes).to(device)
    elif args.classifier == 'conv3':
        model = Conv3AudioClassifier(num_classes=args.num_classes).to(device)
    else:
        raise ValueError(f"Invalid classifier choice: {args.classifier}")
    total_params = compute_model_size(model)
    total_size_mb = total_params / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} M")
    train_dataset = AVHubertFeatureDataset_FromFile(data_dir=args.train_data_dir)
    eval_dataset = AVHubertFeatureDataset_FromFile(data_dir=args.eval_data_dir)

    # if extract featreu online
    #train_dataset = AVHubertFeatureDataset(
    #    ckpt_path=ckpt_path,
    #    manifest_path="data/train.tsv",
    #    label_path="data/train.wrd",
    #    sample_rate=16000,
    #    modalities=["video", "audio"],
    #    normalize=True,
        #)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    trainer.train()

if __name__ == '__main__':
    main()

    """
    max_epoch = cfg.optimization.max_epoch
    for epoch in range(max_epoch):
        # Set the model to training mode
        model.train()

        # Iterate over the training data
        for batch in progress_bar.progress_bar(train_loader):
            features, labels, fids = batch
            # Convert features and labels to the appropriate device
            features, labels = features.to(device), labels.to(device)

            # Prepare the input dictionary expected by Fairseq's trainer
            sample = {
                'net_input': {
                    'src_tokens': features,
                    'src_lengths': torch.tensor([features.size(1)] * features.size(0)),
                },
                'target': labels,
            }

            # Perform a training step
            trainer.train_step(sample)

            # Log or process fids if needed
            print(fids)



# Instantiate the custom datase
ttrain_dataset = AVHubertFeatureDataset(
    ckpt_path=ckpt_path,
    manifest_path="data/train.tsv",
    label_path="data/train.wrd",
    sample_rate=16000,
    modalities=["video", "audio"],
    normalize=True,
)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
train_iter = iter(train_loader)

for batch in progress_bar.progress_bar(train_loader):
    features, labels, fids = batch
    import pdb
    pdb.set_trace()
    
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0]


    # Iterate through the dataset and extract features
    features = []
    labels = []
    fids = []

    for i in tqdm.tqdm(range(len(dataset))):
        data = dataset[i]
        feature, _ = model.extract_finetune(
            source={
                "video": data["video_source"].permute(3, 0, 1, 2).unsqueeze(0),
                "audio": data["audio_source"].unsqueeze(0).permute(0, 2, 1),
            },
            output_layer=None,
        )
        features.append(feature.cpu().detach().numpy())
        labels.append(data["label_list"])  # Adjust this based on how your labels are stored
        fids.append(data["fid"])

    # Save features and labels
    np.save(feat_dir + "/" + ckpt_path + '_features.npy', np.array(features))
    np.save(feat_dir + "/" + ckpt_path + '_labels.npy', np.array(labels))
    np.save(feat_dir + "/" + ckpt_path + '_fids.npy', np.array(fids))
    data = dataset[0]
    print(data["video_source"].transpose(0, -1).unsqueeze(0).shape)
    print(data["audio_source"].unsqueeze(0).permute(0, 2, 1).shape)
    print(data["fid"])
    # print(model.encoder.w2v_model)
    feature, _ = model.extract_finetune(
        source={
            "video": data["video_source"].permute(3, 0, 1, 2).unsqueeze(0),
            "audio": data["audio_source"].unsqueeze(0).permute(0, 2, 1),
        },
        output_layer=None,
    )

    # print(feature.shape)
    
else:
    # if use ft model (with LRS3 + VoxCeleb2)
    import hubert_asr
    raise NotImplementedError


"""
