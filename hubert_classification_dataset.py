# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
import time
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from python_speech_features import logfbank
from scipy.io import wavfile

from torch.utils.data import Dataset
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
import fairseq

DBG = True # if len(sys.argv) == 1 else False

if DBG:
    import utils as custom_utils

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
        stream=sys.stdout,
    )
else:
    from . import utils as custom_utils

logger = logging.getLogger(__name__)


def load_audio_visual(manifest_path, max_keep, min_keep):

    keys = []
    audio_paths = dict()
    video_paths = dict()
    video_sizes = dict()

    with open(manifest_path) as f:
        for line in f:
            items = line.strip().split("\t")
            sz = int(items[-2])  #
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                video_path = items[1]
                audio_path = items[2]
                audio_id = items[0]
                keys.append(audio_id)
                audio_paths[audio_id] = audio_path
                video_paths[audio_id] = video_path
                video_sizes[audio_id] = sz
    tot = len(keys)
    sizes = video_sizes.values()
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return keys, audio_paths, video_paths, tot, video_sizes


def load_label(label_path, tot):
    labels = dict()
    with open(label_path) as f:
        label_lines = [line.rstrip() for line in f]
        assert (
            len(label_lines) == tot
        ), f"number of labels does not match ({len(label_lines)} != {tot})"
        # labels = [labels[i] for i in inds]
        for line in label_lines:
            key, label = line.split("\t")
            labels[key] = int(label)
    return labels


class AVHubertClassificationDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        label_path: str,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        random_crop: bool = False,
        stack_order_audio: int = 4,
        image_mean: float = 0,
        image_std: float = 1,
        image_crop_size: int = 88,
        image_aug: bool = False,
        modalities: Optional[List[str]] = None,
    ):

        self.modalities = set(modalities)
        self.keys, self.audio_paths, self.video_paths, tot, self.video_sizes = (
            load_audio_visual(
                manifest_path,
                max_keep_sample_size,
                min_keep_sample_size,
            )
        )
        self.sample_rate = sample_rate
        self.stack_order_audio = stack_order_audio
        self.shuffle = shuffle
        self.random_crop = random_crop
        self.label_list = None
        if label_path is not None:
            self.label_list = load_label(label_path, tot)
            
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        if image_aug:
            self.transform = custom_utils.Compose(
                [
                    custom_utils.Normalize(0.0, 255.0),
                    custom_utils.RandomCrop((image_crop_size, image_crop_size)),
                    custom_utils.HorizontalFlip(0.5),
                    custom_utils.Normalize(image_mean, image_std),
                ]
            )
        else:
            self.transform = custom_utils.Compose(
                [
                    custom_utils.Normalize(0.0, 255.0),
                    custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                    custom_utils.Normalize(image_mean, image_std),
                ]
            )
        logger.info(f"image transform: {self.transform}")

        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}, "
        )

    def get_labels(self, index):
        label = self.label_list[self.keys[index]]
        #init_label = [0, 0, 0, 0]
        #init_label[label] += 1
        return label

    def load_feature(self, video_path, audio_path):
        """
        Load image and audio feature
        Returns:
        video_feats: numpy.ndarray of shape [T, H, W, 1], audio_feats: numpy.ndarray of shape [T, F]
        """

        def stacker(feats, stack_order):
            """
            Concatenating consecutive audio frames
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(
                -1, stack_order * feat_dim
            )
            return feats

        video_fn, audio_fn = video_path, audio_path
        if "video" in self.modalities:
            video_feats = self.load_video(video_fn)  # [T, H, W, 1]
        else:
            video_feats = None
        if "audio" in self.modalities:
            audio_fn = audio_fn
            sample_rate, wav_data = wavfile.read(audio_fn)
            assert sample_rate == 16_000 and len(wav_data.shape) == 1
            audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(
                np.float32
            )  # [T, F]
            audio_feats = stacker(
                audio_feats, self.stack_order_audio
            )  # [T/stack_order_audio, F*stack_order_audio]
        else:
            audio_feats = None
        if audio_feats is not None and video_feats is not None:
            diff = len(audio_feats) - len(video_feats)
            if diff < 0:
                audio_feats = np.concatenate(
                    [
                        audio_feats,
                        np.zeros(
                            [-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype
                        ),
                    ]
                )
            elif diff > 0:
                audio_feats = audio_feats[:-diff]
        return video_feats, audio_feats

    def load_video(self, audio_name):
        feats = custom_utils.load_video(audio_name)
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats

    def __getitem__(self, index):
        key = self.keys[index]

        video_feats, audio_feats = self.load_feature(
            video_path=self.video_paths[key], audio_path=self.audio_paths[key]
        )
        audio_feats, video_feats = (
            torch.from_numpy(audio_feats.astype(np.float32))
            if audio_feats is not None
            else None
        ), (
            torch.from_numpy(video_feats.astype(np.float32))
            if video_feats is not None
            else None
        )
        if self.normalize and "audio" in self.modalities:
            with torch.no_grad():
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
        if self.label_list is not None:
            labels = self.get_labels(index)
            return {
                "id": index,
                "fid": key,
                "video_source": video_feats,
                "audio_source": audio_feats,
                "label_list": labels,
            }
        else:
            return {
                "id": index,
                "fid": key,
                "video_source": video_feats,
                "audio_source": audio_feats,
            }

    def __len__(self):
        return len(self.keys)

    def crop_to_max_size(self, wav, target_size, start=None):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0
        # longer utterances
        if start is None:
            start, end = 0, target_size
            if self.random_crop:
                start = np.random.randint(0, diff + 1)
                end = size - diff + start
        else:
            end = start + target_size
        return wav[start:end], start

    def collater(self, samples):
        samples = [s for s in samples if s["id"] is not None]
        if len(samples) == 0:
            return {}

        audio_source, video_source = [s["audio_source"] for s in samples], [
            s["video_source"] for s in samples
        ]
        if audio_source[0] is None:
            audio_source = None
        if video_source[0] is None:
            video_source = None
        if audio_source is not None:
            audio_sizes = [len(s) for s in audio_source]
        else:
            audio_sizes = [len(s) for s in video_source]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        if audio_source is not None:
            collated_audios, padding_mask, audio_starts = self.collater_audio(
                audio_source, audio_size
            )
        else:
            collated_audios, audio_starts = None, None
        if video_source is not None:
            collated_videos, padding_mask, audio_starts = self.collater_audio(
                video_source, audio_size, audio_starts
            )
        else:
            collated_videos = None
        targets_by_label = [s["label_list"] for s in samples]
        source = {"audio": collated_audios, "video": collated_videos}
        net_input = {"source": source, "padding_mask": padding_mask}
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
            "utt_id": [s["fid"] for s in samples],
        }

        batch["target_lengths"] = [1] * len(samples)
        batch["ntokens"] = [1] * len(samples)

        return batch

    def collater_audio(self, audios, audio_size, audio_starts=None):
        audio_feat_shape = list(audios[0].shape[1:])
        collated_audios = audios[0].new_zeros(
            [len(audios), audio_size] + audio_feat_shape
        )
        padding_mask = torch.BoolTensor(len(audios), audio_size).fill_(False)  #
        start_known = audio_starts is not None
        audio_starts = [0 for _ in audios] if not start_known else audio_starts
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat(
                    [audio, audio.new_full([-diff] + audio_feat_shape, 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size, audio_starts[i] if start_known else None
                )
        if len(audios[0].shape) == 2:
            collated_audios = collated_audios.transpose(1, 2)  # [B, T, F] -> [B, F, T]
        else:
            collated_audios = collated_audios.permute(
                (0, 4, 1, 2, 3)
            ).contiguous()  # [B, T, H, W, C] -> [B, C, T, H, W]
        return collated_audios, padding_mask, audio_starts

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.video_sizes[index]
        return min(self.video_sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.video_sizes)
        return np.lexsort(order)[::-1]


class AVHubertFeatureDataset(Dataset):
    def __init__(self, ckpt_path, manifest_path, label_path, sample_rate, modalities, normalize=True, device=None):
        import hubert_pretraining, hubert
        self.models, self.cfg, self.task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = self.models[0]
        self.dataset = AVHubertClassificationDataset(
            manifest_path=manifest_path,
            label_path=label_path,
            sample_rate=sample_rate,
            modalities=modalities,
            normalize=normalize,
        )
        # Move the model to GPU if available
        
        self.model.to(device)
        self.device = device
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        video_tensor = data["video_source"].permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        audio_tensor = data["audio_source"].unsqueeze(0).permute(0, 2, 1).to(self.device)
        
        with torch.no_grad():
            feature, _ = self.model.extract_finetune(
                source={"video": video_tensor, "audio": audio_tensor},
                output_layer=None,
            )
        label = -1
        if "label_list" in data:
            label = data["label_list"]
        fid = data["fid"]
        return feature.squeeze(0), label, fid  # Remove the batch dimension from feature


class AVHubertFeatureDataset_FromFile(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        # Dictionary to store file paths indexed by keys
        self.data_dict = {}

        # Process feature files
        feature_files = [f for f in os.listdir(data_dir) if f.startswith('feature_')]
        for f in feature_files:
            key = f[len('feature_'):].rsplit('.', 1)[0]
            self.data_dict[key] = {'feature': os.path.join(data_dir, f)}
        
        # Process label files
        label_files = [f for f in os.listdir(data_dir) if f.startswith('label_')]
        for f in label_files:
            key = f[len('label_'):].rsplit('.', 1)[0]
            if key in self.data_dict:
                self.data_dict[key]['label'] = os.path.join(data_dir, f)
        
        # Process fid files
        fid_files = [f for f in os.listdir(data_dir) if f.startswith('fid_')]
        for f in fid_files:
            key = f[len('fid_'):].rsplit('.', 1)[0]
            if key in self.data_dict:
                self.data_dict[key]['fid'] = os.path.join(data_dir, f)
        
        # Filter out incomplete entries
        self.data_keys = [key for key in self.data_dict if 'label' in self.data_dict[key] and 'fid' in self.data_dict[key]]

    def __len__(self):
        return len(self.data_keys)
    
    def __getitem__(self, idx):
        key = self.data_keys[idx]
        feature_file = self.data_dict[key]['feature']
        label_file = self.data_dict[key]['label']
        fid_file = self.data_dict[key]['fid']
        
        feature = torch.from_numpy(np.load(feature_file))
        label = torch.from_numpy(np.load(label_file))
        fid = np.load(fid_file).item()
        
        return feature, label, fid
    
