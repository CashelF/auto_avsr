import os

import torch
from pytorch_lightning import LightningDataModule

from .av_dataset import AVDataset
from .transforms import AudioTransform, VideoTransform


def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    elif len(samples[0].shape) == 2:
        pass  # collated_batch: [B, T, 1]
    elif len(samples[0].shape) == 4:
        pass  # collated_batch: [B, T, C, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        pad_val = -1 if data_type == "target" else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out


def _batch_by_token_count(idx_target_lengths, max_frames, batch_size=None):
    batches = []
    current_batch = []
    current_token_count = 0
    for idx, target_length in idx_target_lengths:
        if current_token_count + target_length > max_frames or (
            batch_size and len(current_batch) == batch_size
        ):
            batches.append(current_batch)
            current_batch = [idx]
            current_token_count = target_length
        else:
            current_batch.append(idx)
            current_token_count += target_length

    if current_batch:
        batches.append(current_batch)

    return batches


class CustomBucketDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset, lengths, max_frames, num_buckets, shuffle=False, batch_size=None
    ):
        super().__init__()

        assert len(dataset) == len(lengths)

        self.dataset = dataset

        max_length = max(lengths)
        min_length = min(lengths)

        assert max_frames >= max_length

        buckets = torch.linspace(min_length, max_length, num_buckets)
        lengths = torch.tensor(lengths)
        bucket_assignments = torch.bucketize(lengths, buckets)

        idx_length_buckets = [
            (idx, length, bucket_assignments[idx]) for idx, length in enumerate(lengths)
        ]
        if shuffle:
            idx_length_buckets = random.sample(
                idx_length_buckets, len(idx_length_buckets)
            )
        else:
            idx_length_buckets = sorted(
                idx_length_buckets, key=lambda x: x[1], reverse=True
            )
        sorted_idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[2])
        self.batches = _batch_by_token_count(
            [(idx, length) for idx, length, _ in sorted_idx_length_buckets],
            max_frames,
            batch_size=batch_size,
        )

    def __getitem__(self, idx):
        return [self.dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)


class DataModule(LightningDataModule):
    def __init__(
        self,
        args=None,
        batch_size=None,
        train_num_buckets=50,
        train_shuffle=True,
        num_workers=10,
    ):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.train_num_buckets = train_num_buckets
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers

    def train_dataloader(self):
        train_label_path = self._resolve_label_path(self.args.root_dir, self.args.train_file)
        dataset = AVDataset(
            root_dir=self.args.root_dir,
            label_path=train_label_path,
            subset="train",
            modality=self.args.modality,
            audio_transform=AudioTransform("train"),
            video_transform=VideoTransform("train"),
        )
        dataset = CustomBucketDataset(
            dataset,
            dataset.input_lengths,
            self.args.max_frames,
            self.train_num_buckets,
            batch_size=self.batch_size,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=None,
            shuffle=self.train_shuffle,
            collate_fn=collate_pad,
        )
        return dataloader

    def val_dataloader(self):
        val_label_path = self._resolve_label_path(self.args.root_dir, self.args.val_file)
        dataset = AVDataset(
            root_dir=self.args.root_dir,
            label_path=val_label_path,
            subset="val",
            modality=self.args.modality,
            audio_transform=AudioTransform("val"),
            video_transform=VideoTransform("val"),
        )
        dataset = CustomBucketDataset(
            dataset, dataset.input_lengths, 1000, 1, batch_size=self.batch_size
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=collate_pad,
        )
        return dataloader

    def test_dataloader(self):
        test_label_path = self._resolve_label_path(self.args.root_dir, self.args.test_file)
        dataset = AVDataset(
            root_dir=self.args.root_dir,
            label_path=test_label_path,
            subset="test",
            modality=self.args.modality,
            audio_transform=AudioTransform(
                "test", snr_target=self.args.decode_snr_target
            ),
            video_transform=VideoTransform("test"),
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        return dataloader

    def _resolve_label_path(self, root_dir, label_file):
        """Match preprocessing output layout by supporting label files at root or in labels/."""

        # Absolute path provided.
        if os.path.isabs(label_file):
            if os.path.exists(label_file):
                return label_file
            raise FileNotFoundError(f"Label file not found: {label_file}")

        # Prefer root_dir/<file> when the preprocessing script emits lists there.
        direct_path = os.path.join(root_dir, label_file)
        if os.path.exists(direct_path):
            return direct_path

        # Fall back to root_dir/labels/<file> for older layouts.
        labels_path = os.path.join(root_dir, "labels", label_file)
        if os.path.exists(labels_path):
            return labels_path

        raise FileNotFoundError(
            f"Could not locate label file '{label_file}' in '{root_dir}' or '{os.path.join(root_dir, 'labels')}'."
        )
