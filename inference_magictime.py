import os
import json
import time
import torch
import random
import inspect
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available

from utils.unet import UNet3DConditionModel
from utils.pipeline_magictime import MagicTimePipeline
from utils.util import save_videos_grid
from utils.util import load_weights

@torch.no_grad()
def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    if 'counter' not in globals():
        globals()['counter'] = 0
    unique_id = globals()['counter']
    globals()['counter'] += 1
    savedir_base = f"{Path(args.config).stem}"
    savedir_prefix = "outputs"
    savedir = None
    if args.save_path:
        savedir = os.path.join(savedir_prefix, args.save_path, f"{savedir_base}-{unique_id}")
    else:
        savedir = os.path.join(savedir_prefix, f"{savedir_base}-{unique_id}")
    while os.path.exists(savedir):
        unique_id = globals()['counter']
        globals()['counter'] += 1
        if args.save_path:
            savedir = os.path.join(savedir_prefix, args.save_path, f"{savedir_base}-{unique_id}")
        else:
            savedir = os.path.join(savedir_prefix, f"{savedir_base}-{unique_id}")
    os.makedirs(savedir)
    print(f"The results will be save to {savedir}")

    model_config = OmegaConf.load(args.config)[0]
    inference_config = OmegaConf.load(args.config)[1]

    if model_config.magic_adapter_s_path:
        print("Use MagicAdapter-S")
    if model_config.magic_adapter_t_path:
        print("Use MagicAdapter-T")
    if model_config.magic_text_encoder_path:
        print("Use Magic_Text_Encoder")

    samples = []

    # create validation pipeline
    tokenizer    = CLIPTokenizer.from_pretrained(model_config.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_config.pretrained_model_path, subfolder="text_encoder").cuda()
    vae          = AutoencoderKL.from_pretrained(model_config.pretrained_model_path, subfolder="vae").cuda()
    unet = UNet3DConditionModel.from_pretrained_2d(model_config.pretrained_model_path, subfolder="unet",
                                                   unet_additional_kwargs=OmegaConf.to_container(
                                                       inference_config.unet_additional_kwargs)).cuda()

    # set xformers
    if is_xformers_available() and (not args.without_xformers):
        unet.enable_xformers_memory_efficient_attention()

    pipeline = MagicTimePipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
    ).to("cuda")

    pipeline = load_weights(
        pipeline,
        motion_module_path=model_config.get("motion_module", ""),
        dreambooth_model_path=model_config.get("dreambooth_path", ""),
        magic_adapter_s_path=model_config.get("magic_adapter_s_path", ""),
        magic_adapter_t_path=model_config.get("magic_adapter_t_path", ""),
        magic_text_encoder_path=model_config.get("magic_text_encoder_path", ""),
    ).to("cuda")

    sample_idx = 0
    if args.human:
        sample_idx = 0  # Initialize sample index
        while True:
            user_prompt = input("Enter your prompt (or type 'exit' to quit): ")
            if user_prompt.lower() == "exit":
                break

            random_seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
            torch.manual_seed(random_seed)

            print(f"current seed: {random_seed}")
            print(f"sampling {user_prompt} ...")

            # Now, you directly use `user_prompt` to generate a video.
            # The following is a placeholder call; you need to adapt it to your actual video generation function.
            sample = pipeline(
                user_prompt,
                num_inference_steps=model_config.steps,
                guidance_scale=model_config.guidance_scale,
                width=model_config.W,
                height=model_config.H,
                video_length=model_config.L,
            ).videos

            # Adapt the filename to avoid conflicts and properly represent the content
            prompt_for_filename = "-".join(user_prompt.replace("/", "").split(" ")[:10])
            save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{random_seed}-{prompt_for_filename}.gif")
            print(f"save to {savedir}/sample/{sample_idx}-{random_seed}-{prompt_for_filename}.gif")

            sample_idx += 1
    elif args.run_csv:
        print("run_csv")
        file_path = args.run_csv
        data = pd.read_csv(file_path)
        for index, row in data.iterrows():
            user_prompt = row['name']  # Set the user_prompt to the 'name' field of the current row
            videoid = row['videoid']  # Extract videoid for filename

            random_seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
            torch.manual_seed(random_seed)

            print(f"current seed: {random_seed}")
            print(f"sampling {user_prompt} ...")

            sample = pipeline(
                user_prompt,
                num_inference_steps=model_config.steps,
                guidance_scale=model_config.guidance_scale,
                width=model_config.W,
                height=model_config.H,
                video_length=model_config.L,
            ).videos

            # Adapt the filename to avoid conflicts and properly represent the content
            save_videos_grid(sample, f"{savedir}/sample/{videoid}.gif")
            print(f"save to {savedir}/sample/{videoid}.gif")
    elif args.run_json:
        print("run_json")
        file_path = args.run_json

        with open(file_path, 'r') as file:
            data = json.load(file)

        prompts = []
        videoids = []
        senids = []

        for item in data:
            prompts.append(item['caption'])
            videoids.append(item['video_id'])
            senids.append(item['sen_id'])

        n_prompts = list(model_config.n_prompt) * len(prompts) if len(
            model_config.n_prompt) == 1 else model_config.n_prompt

        random_seeds = model_config.get("seed", [-1])
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds

        model_config.random_seed = []
        for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
            filename = f"MSRVTT/sample/{videoids[prompt_idx]}-{senids[prompt_idx]}.gif"

            if os.path.exists(filename):
                print(f"File {filename} already exists, skipping...")
                continue

            # manually set random seed for reproduction
            if random_seed != -1:
                torch.manual_seed(random_seed)
            else:
                torch.seed()
            model_config.random_seed.append(torch.initial_seed())

            print(f"current seed: {torch.initial_seed()}")
            print(f"sampling {prompt} ...")

            sample = pipeline(
                prompt,
                num_inference_steps=model_config.steps,
                guidance_scale=model_config.guidance_scale,
                width=model_config.W,
                height=model_config.H,
                video_length=model_config.L,
            ).videos

            # Adapt the filename to avoid conflicts and properly represent the content
            save_videos_grid(sample, filename)
            print(f"save to {filename}")
    else:
        prompts = model_config.prompt
        n_prompts = list(model_config.n_prompt) * len(prompts) if len(
            model_config.n_prompt) == 1 else model_config.n_prompt

        random_seeds = model_config.get("seed", [-1])
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds

        model_config.random_seed = []
        for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):

            # manually set random seed for reproduction
            if random_seed != -1:
                torch.manual_seed(random_seed)
                np.random.seed(random_seed)
                random.seed(random_seed)
            else:
                torch.seed()
            model_config.random_seed.append(torch.initial_seed())

            print(f"current seed: {torch.initial_seed()}")
            print(f"sampling {prompt} ...")
            sample = pipeline(
                prompt,
                negative_prompt=n_prompt,
                num_inference_steps=model_config.steps,
                guidance_scale=model_config.guidance_scale,
                width=model_config.W,
                height=model_config.H,
                video_length=model_config.L,
            ).videos
            samples.append(sample)

            prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
            save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{random_seed}-{prompt}.gif")
            print(f"save to {savedir}/sample/{random_seed}-{prompt}.gif")

            sample_idx += 1
        samples = torch.concat(samples)
        save_videos_grid(samples, f"{savedir}/merge_all.gif", n_rows=4)

    OmegaConf.save(model_config, f"{savedir}/model_config.yaml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--without-xformers", action="store_true")
    parser.add_argument("--human", action="store_true", help="Enable human mode for interactive video generation")
    parser.add_argument("--run-csv", type=str, default=None)
    parser.add_argument("--run-json", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)

    args = parser.parse_args()
    main(args)
