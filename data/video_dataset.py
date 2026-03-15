"""
Video dataset that reads a JSON annotation file and loads video frames.

Expected JSON format (list of dicts):
[
    {
        "id": 0,
        "video": "E2E_VP_default/val/down/circle_blue_001.mp4",
        "question": "...",
        ...
    },
    ...
]

The "video" field is a relative path joined with video_root_dir.
Other fields are preserved as metadata.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu


@dataclass
class VideoSample:
    """A single video sample with metadata."""
    id: Any
    video_path: str           # absolute path
    frames: np.ndarray        # (num_frames, H, W, 3)
    num_frames: int
    video_time: float         # total video duration in seconds
    frame_time: str           # comma-separated timestamps, e.g. "0.00s,1.23s,..."
    metadata: Dict[str, Any]  # all other fields from JSON (question, answer, etc.)


class VideoFrameDataset(Dataset):
    """
    Dataset that loads video frames from a JSON annotation file.

    Usage:
        dataset = VideoFrameDataset(
            json_path="annotations.json",
            video_root_dir="/data/videos",
            num_frames=8,
        )
        sample = dataset[0]  # VideoSample

    For feature extraction with an encoder, use make_dataloader() which
    handles preprocessing and batching.

    Args:
        json_path: Path to JSON annotation file (list of dicts).
        video_root_dir: Root directory to join with the "video" field.
        num_frames: Number of frames to uniformly sample per video.
        video_key: Key in JSON dict for video relative path. Default: "video".
        id_key: Key in JSON dict for sample ID. Default: "id".
        force_sample: Always use uniform sampling even if video has fewer frames.
    """

    def __init__(
        self,
        json_path: str,
        video_root_dir: str,
        num_frames: int = 8,
        video_key: str = "video",
        id_key: str = "id",
        force_sample: bool = False,
    ):
        self.video_root_dir = video_root_dir
        self.num_frames = num_frames
        self.video_key = video_key
        self.id_key = id_key
        self.force_sample = force_sample

        # Load annotations
        with open(json_path, "r") as f:
            self.annotations = json.load(f)

        print(f"[VideoFrameDataset] Loaded {len(self.annotations)} samples from {json_path}")
        print(f"  - video_root_dir: {video_root_dir}")
        print(f"  - num_frames: {num_frames}")

    def __len__(self) -> int:
        return len(self.annotations)

    def _sample_frames(self, video_path: str) -> tuple:
        """
        Sample frames from video using Decord.
        Follows LLaVA-NeXT's process_video_with_decord logic.

        Returns:
            frames: np.ndarray of shape (num_frames, H, W, 3)
            video_time: float, total video duration in seconds
            frame_time: str, comma-separated timestamps
            num_frames_sampled: int
        """
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        avg_fps = vr.get_avg_fps()
        video_time = total_frame_num / avg_fps

        # Build initial frame indices (1 frame per second)
        fps_stride = max(1, round(avg_fps))
        frame_idx = list(range(0, total_frame_num, fps_stride))
        frame_time_list = [i / avg_fps for i in frame_idx]

        # Uniform sampling if too many frames or force_sample
        if self.num_frames > 0:
            if len(frame_idx) > self.num_frames or self.force_sample:
                uniform_idx = np.linspace(0, total_frame_num - 1, self.num_frames, dtype=int)
                frame_idx = uniform_idx.tolist()
                frame_time_list = [i / avg_fps for i in frame_idx]

        frames = vr.get_batch(frame_idx).asnumpy()  # (N, H, W, 3)
        frames = np.stack(frames)

        frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time_list])

        # https://github.com/dmlc/decord/issues/208
        vr.seek(0)

        return frames, video_time, frame_time_str, len(frame_idx)

    def __getitem__(self, idx: int) -> VideoSample:
        ann = self.annotations[idx]

        # Build absolute video path
        rel_path = ann[self.video_key]
        video_path = os.path.join(self.video_root_dir, rel_path)

        # Sample frames
        frames, video_time, frame_time, num_frames = self._sample_frames(video_path)

        # Collect metadata (everything except video path and id)
        metadata = {k: v for k, v in ann.items() if k not in (self.video_key, self.id_key)}

        return VideoSample(
            id=ann.get(self.id_key, idx),
            video_path=video_path,
            frames=frames,
            num_frames=num_frames,
            video_time=video_time,
            frame_time=frame_time,
            metadata=metadata,
        )


def collate_video_samples(
    samples: List[VideoSample],
    image_processor,
) -> dict:
    """
    Collate function for DataLoader.

    Preprocesses raw frames through the vision encoder's image_processor,
    then stacks into a batch.

    Args:
        samples: List of VideoSample from dataset
        image_processor: The encoder's image preprocessor (e.g. SigLipImageProcessor)

    Returns:
        dict with keys:
            - "pixel_values": Tensor (total_frames, C, H, W)
            - "frame_counts": List[int], number of frames per video (for splitting later)
            - "ids": List[Any], sample IDs
            - "video_paths": List[str]
            - "metadata": List[Dict]
    """
    all_pixel_values = []
    frame_counts = []
    ids = []
    video_paths = []
    metadata_list = []

    for sample in samples:
        # image_processor expects list of numpy arrays or PIL images
        # sample.frames is (num_frames, H, W, 3) numpy
        processed = image_processor.preprocess(
            sample.frames,  # numpy (N, H, W, 3) — each frame treated as an image
            return_tensors="pt",
        )
        pixel_values = processed["pixel_values"]  # (N, C, H, W)

        all_pixel_values.append(pixel_values)
        frame_counts.append(sample.num_frames)
        ids.append(sample.id)
        video_paths.append(sample.video_path)
        metadata_list.append(sample.metadata)

    return {
        "pixel_values": torch.cat(all_pixel_values, dim=0),  # (total_frames, C, H, W)
        "frame_counts": frame_counts,
        "ids": ids,
        "video_paths": video_paths,
        "metadata": metadata_list,
    }


def make_dataloader(
    json_path: str,
    video_root_dir: str,
    image_processor,
    num_frames: int = 8,
    batch_size: int = 1,
    num_workers: int = 4,
    video_key: str = "video",
    id_key: str = "id",
    force_sample: bool = False,
) -> DataLoader:
    """
    Convenience function to create a DataLoader for video feature extraction.

    Args:
        json_path: Path to JSON annotation file.
        video_root_dir: Root directory for video files.
        image_processor: Vision encoder's image preprocessor.
        num_frames: Frames to sample per video.
        batch_size: Number of videos per batch.
        num_workers: DataLoader workers.
        video_key: JSON key for video path.
        id_key: JSON key for sample ID.
        force_sample: Always uniform sample.

    Returns:
        DataLoader yielding batched dicts from collate_video_samples.

    Example:
        encoder = build_vision_encoder("siglip", "google/siglip-so400m-patch14-384")
        loader = make_dataloader(
            json_path="annotations.json",
            video_root_dir="/data/videos",
            image_processor=encoder.get_image_processor(),
            num_frames=8,
            batch_size=4,
        )
        for batch in loader:
            features = encoder.encode_images(batch["pixel_values"])
            # features: (total_frames, 729, 1152)
            # split by batch["frame_counts"] to get per-video features
    """
    dataset = VideoFrameDataset(
        json_path=json_path,
        video_root_dir=video_root_dir,
        num_frames=num_frames,
        video_key=video_key,
        id_key=id_key,
        force_sample=force_sample,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda samples: collate_video_samples(samples, image_processor),
        pin_memory=True,
    )

    return loader
