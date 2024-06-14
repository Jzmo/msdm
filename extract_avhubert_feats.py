import argparse
import os
import tqdm
import numpy as np
import torch
from hubert_classification_dataset import AVHubertFeatureDataset

def main(args):
    # Assuming you have already defined AVHubertFeatureDataset class

    # Define paths for saving .npy files
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = AVHubertFeatureDataset(
        ckpt_path=args.ckpt_path,
        manifest_path=args.manifest_path,
        label_path=args.label_path,
        sample_rate=args.sample_rate,
        modalities=args.modalities,
        normalize=args.normalize,
        device=device,
    )

    # Iterate over the dataset and save features, labels, and file IDs
    for i in tqdm.tqdm(range(len(train_dataset))):
        feature, label, fid = train_dataset[i]
        feature_path = os.path.join(args.output_dir, f'feature_{fid}.npy')
        label_path = os.path.join(args.output_dir, f'label_{fid}.npy')
        fid_path = os.path.join(args.output_dir, f'fid_{fid}.npy')
        np.save(feature_path, feature.cpu().detach().numpy())
        np.save(label_path, np.array(label))
        np.save(fid_path, np.array(fid))

    print("Features, labels, and file IDs have been saved to .npy files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments for dataset processing")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/base_vox_iter5.pt", help="Path to the checkpoint")
    parser.add_argument("--manifest_path", type=str, default="data/train.tsv", help="Path to the manifest file")
    parser.add_argument("--label_path", type=str, default=None, help="Path to the label file")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate of the audio")
    parser.add_argument("--modalities", nargs="+", type=str, default=["video", "audio"], help="Modalities to include")
    parser.add_argument("--normalize", action="store_true", help="Normalize the features")
    parser.add_argument("--output_dir", type=str, default="feats/train", help="Directory to save .npy files")

    args = parser.parse_args()
    main(args)

