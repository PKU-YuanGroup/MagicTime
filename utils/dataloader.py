import os
import csv
import decord
import random
import numpy as np
from tqdm import tqdm
from typing import Optional

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from accelerate.logging import get_logger

logger = get_logger(__name__)

class VideoDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[str] = None, 
        video_folder: Optional[str] = None,
        height: int = 480,
        width: int = 720,
        fps: int = 8,
        max_num_frames: int = 49,
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        cache_dir: Optional[str] = None,
        id_token: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.csv_path = csv_path
        self.video_folder = video_folder
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
    
        self.height = height
        self.width = width
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.cache_dir = cache_dir
        self.id_token = id_token or ""

        self.instance_prompts, self.instance_video_paths, self.instance_transmit = self._load_dataset_from_local_path()

        self.instance_prompts = [self.id_token + prompt for prompt in self.instance_prompts]

        self.num_instance_videos = len(self.instance_video_paths)
        if self.num_instance_videos != len(self.instance_prompts):
            raise ValueError(
                f"Expected length of instance prompts and videos to be the same but found {len(self.instance_prompts)=} and {len(self.instance_video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        # self.instance_videos = self._preprocess_data()

    def __len__(self):
        return self.num_instance_videos

    def __getitem__(self, index):
        decord.bridge.set_bridge("torch")

        train_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
            ]
        )

        video_reader = decord.VideoReader(uri=self.instance_video_paths[index], width=self.width, height=self.height)
        video_num_frames = len(video_reader)

        indices = self._generate_frame_indices(video_length=video_num_frames, n_frames=self.max_num_frames, sample_stride=4, is_transmit=0)
        frames = video_reader.get_batch(indices)

        # start_frame = min(self.skip_frames_start, video_num_frames)
        # end_frame = max(0, video_num_frames - self.skip_frames_end)
        # if end_frame <= start_frame:
        #     frames = video_reader.get_batch([start_frame])
        # elif end_frame - start_frame <= self.max_num_frames:
        #     frames = video_reader.get_batch(list(range(start_frame, end_frame)))
        # else:
        #     indices = list(range(start_frame, end_frame, (end_frame - start_frame) // self.max_num_frames))
        #     frames = video_reader.get_batch(indices)

        # # Ensure that we don't go over the limit
        # frames = frames[: self.max_num_frames]
        # selected_num_frames = frames.shape[0]

        # # Choose first (4k + 1) frames as this is how many is required by the VAE
        # remainder = (3 + (selected_num_frames % 4)) % 4
        # if remainder != 0:
        #     frames = frames[:-remainder]

        selected_num_frames = frames.shape[0]
        assert (selected_num_frames - 1) % 4 == 0

        # Training transforms
        frames = frames.float()
        frames = torch.stack([train_transforms(frame) for frame in frames], dim=0)

        return {
            "prompts": self.instance_prompts[index],
            "videos": frames.permute(0, 3, 1, 2).contiguous(),  # [F, C, H, W]
        }

    def _get_frame_indices_adjusted(self, video_length, n_frames):
        indices = list(range(video_length))
        additional_frames_needed = n_frames - video_length
        
        repeat_indices = []
        for i in range(additional_frames_needed):
            index_to_repeat = i % video_length
            repeat_indices.append(indices[index_to_repeat])
        
        all_indices = indices + repeat_indices
        all_indices.sort()

        return all_indices

    def _generate_frame_indices(self, video_length, n_frames, sample_stride, is_transmit):
        prob_execute_original = 1 if int(is_transmit) == 0 else 0

        # Generate a random number to decide which block of code to execute
        if random.random() < prob_execute_original:
            if video_length <= n_frames:
                return self._get_frame_indices_adjusted(video_length, n_frames)
            else:
                interval = (video_length - 1) / (n_frames - 1)
                indices = [int(round(i * interval)) for i in range(n_frames)]
                indices[-1] = video_length - 1
                return indices
        else:
            if video_length <= n_frames:
                return self._get_frame_indices_adjusted(video_length, n_frames)
            else:
                clip_length = min(video_length, (n_frames - 1) * sample_stride + 1)
                start_idx = random.randint(0, video_length - clip_length)
                return np.linspace(start_idx, start_idx + clip_length - 1, n_frames, dtype=int).tolist()

    def _load_dataset_from_local_path(self):
        instance_videos = []
        instance_prompts = []
        instance_transmit = []

        for video_dict in self.dataset:
            videoid = video_dict['videoid']
            video_path = os.path.join(self.video_folder, f"{videoid}.mp4")
            if not os.path.exists(video_path):
                continue
            instance_videos.append(video_path)
            instance_prompts.append(video_dict['name'])
            instance_transmit.append(video_dict['is_transmit'])

        return instance_prompts, instance_videos, instance_transmit

    def _preprocess_data(self):
        try:
            import decord
        except ImportError:
            raise ImportError(
                "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
            )

        decord.bridge.set_bridge("torch")

        videos = []
        train_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
            ]
        )

        for filename, is_transmit in tqdm(zip(self.instance_video_paths, self.instance_transmit)):
            try:
                video_reader = decord.VideoReader(uri=filename, width=self.width, height=self.height)
                video_num_frames = len(video_reader)

                indices = self._generate_frame_indices(video_length=video_num_frames, n_frames=self.max_num_frames, sample_stride=4, is_transmit=0)
                frames = video_reader.get_batch(indices)

                # start_frame = min(self.skip_frames_start, video_num_frames)
                # end_frame = max(0, video_num_frames - self.skip_frames_end)
                # if end_frame <= start_frame:
                #     frames = video_reader.get_batch([start_frame])
                # elif end_frame - start_frame <= self.max_num_frames:
                #     frames = video_reader.get_batch(list(range(start_frame, end_frame)))
                # else:
                #     indices = list(range(start_frame, end_frame, (end_frame - start_frame) // self.max_num_frames))
                #     frames = video_reader.get_batch(indices)

                # # Ensure that we don't go over the limit
                # frames = frames[: self.max_num_frames]
                # selected_num_frames = frames.shape[0]

                # # Choose first (4k + 1) frames as this is how many is required by the VAE
                # remainder = (3 + (selected_num_frames % 4)) % 4
                # if remainder != 0:
                #     frames = frames[:-remainder]

                selected_num_frames = frames.shape[0]
                assert (selected_num_frames - 1) % 4 == 0

                # Training transforms
                frames = frames.float()
                frames = torch.stack([train_transforms(frame) for frame in frames], dim=0)
                videos.append(frames.permute(0, 3, 1, 2).contiguous())  # [F, C, H, W]
            except:
                print("error dataloader")
                continue

        return videos
    
if __name__ == "__main__":
    train_dataset = VideoDataset(
        csv_path="/storage/ysh/Ckpts/ChronoMagic/caption/ChronoMagic_train.csv", 
        video_folder="/storage/ysh/Ckpts/ChronoMagic/video",
        height=480,
        width=720,
        fps=8,
        max_num_frames=49,
        skip_frames_start=0,
        skip_frames_end=0,
    )