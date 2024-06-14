import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from hubert_classification_dataset import AVHubertFeatureDataset_FromFile
from dysarthria_model import Conv3AudioClassifier, ResNet50AudioClassifier
from train_dysarthria import collate_fn

def predict(args):
    # Check if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Predicting on {device}")

    # Define the model architecture based on the argument
    if args.classifier == 'resnet50':
        model = ResNet50AudioClassifier(num_classes=args.num_classes).to(device)
    elif args.classifier == 'conv3':
        model = Conv3AudioClassifier(num_classes=args.num_classes).to(device)
    else:
        raise ValueError(f"Invalid classifier choice: {args.classifier}")

    # Load the model from checkpoint
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)

    # Set the model to evaluation mode
    model.eval()

    # Prepare your test dataset
    test_dataset = AVHubertFeatureDataset_FromFile(data_dir=args.test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=collate_fn)

    all_predictions = []

    # Make predictions
    with torch.no_grad():
        with open(args.output_file, 'w') as f:
            for batch in test_loader:
                features, fids = batch["features"], batch["fids"]
                features = features.to(device)
                outputs = model(features, None, fids)
                preds = outputs.logits.argmax(-1)
                for fid, pred in zip(fids, preds):
                    f.write(f"{fid} {pred}\n")

    print(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with a trained audio classifier")

    # Add arguments
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--test_data_dir', type=str, required=True, help='Directory containing test data')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=64, help='Batch size per device for evaluation')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes for the classifier')
    parser.add_argument('--classifier', type=str, default='conv3', choices=['resnet50', 'conv3'], help='Choose the classifier architecture')
    parser.add_argument('--output_file', type=str, required=True, help='File to save predictions')

    # Parse arguments
    args = parser.parse_args()

    predict(args)
