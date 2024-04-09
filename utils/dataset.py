import os, csv, random
import numpy as np
from decord import VideoReader
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class ChronoMagic(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=512, sample_stride=4, sample_n_frames=16,
            is_image=False,
            is_uniform=True,
        ):
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)

        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        self.is_uniform      = is_uniform
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

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
        
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, name, is_transmit = video_dict['videoid'], video_dict['name'], video_dict['is_transmit']
        
        video_dir    = os.path.join(self.video_folder, f"{videoid}.mp4")
        video_reader = VideoReader(video_dir, num_threads=0)
        video_length = len(video_reader)

        batch_index = self._generate_frame_indices(video_length, self.sample_n_frames, self.sample_stride, is_transmit) if not self.is_image else [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2) / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]
        
        return pixel_values, name, videoid

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name, videoid = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name, id=videoid)
        return sample