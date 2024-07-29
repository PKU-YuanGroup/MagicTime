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
    savedir = None
    savedir = os.path.join(args.save_path, f"{unique_id}")
    while os.path.exists(savedir):
        unique_id = globals()['counter']
        globals()['counter'] += 1
        savedir = os.path.join(args.save_path, f"{unique_id}")
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

    if args.human:
        sample_idx = 0
        while True:
            user_prompt = input("Enter your prompt (or type 'exit' to quit): ")
            if user_prompt.lower() == "exit":
                break

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

            prompt_for_filename = "-".join(user_prompt.replace("/", "").split(" ")[:10])
            save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{random_seed}-{prompt_for_filename}.gif")
            print(f"save to {savedir}/sample/{sample_idx}-{random_seed}-{prompt_for_filename}.gif")

            sample_idx += 1
    else:
        default = True
        batch_size = args.batch_size

        if args.run_csv:
            print("run csv")
            default = False
            file_path = args.run_csv
            data = pd.read_csv(file_path)
            prompts = data['name'].tolist()
            videoids = data['videoid'].tolist()
        elif args.run_json:
            print("run json")
            default = False
            file_path = args.run_json
            with open(file_path, 'r') as file:
                data = json.load(file)
            prompts = []
            videoids = []
            senids = []
            for item in data['sentences']:
                prompts.append(item['caption'])
                videoids.append(item['video_id'])
                senids.append(item['sen_id'])
        elif args.run_txt:
            print("run txt")
            default = False
            file_path = args.run_txt
            with open(file_path, 'r') as file:
                prompts = [line.strip() for line in file.readlines()]
            videoids = [f"video_{i}" for i in range(len(prompts))]
        else:
            prompts = model_config.prompt
            videoids = [f"video_{i}" for i in range(len(prompts))]

        os.makedirs(savedir, exist_ok=True)

        for i in range(0, len(prompts), batch_size):
            batch_prompts_raw = prompts[i : i + batch_size]
            batch_prompts = [prompt for prompt in batch_prompts_raw]

            if args.run_csv or args.run_json or args.run_txt or default:
                batch_videoids = videoids[i : i + batch_size]
            if args.run_json:
                batch_senids = senids[i : i + batch_size]

            flag = True
            for idx in range(len(batch_prompts)):
                if args.run_csv or args.run_txt or default:
                    new_filename = f"{batch_videoids[idx]}.mp4"
                if args.run_json:
                    new_filename = f"{batch_videoids[idx]}-{batch_senids[idx]}.mp4"
                if not os.path.exists(os.path.join(savedir, new_filename)):
                    flag = False
                    break
            if flag:
                print("skipping")
                continue

            n_prompts  = list(model_config.n_prompt) * len(batch_prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
                
            random_seed = torch.randint(0, 2**32 - 1, (1,)).item()
            torch.manual_seed(random_seed)

            print(f"current seed: {random_seed}")

            results = pipeline(
                batch_prompts,
                negative_prompt     = n_prompts,
                num_inference_steps = model_config.steps,
                guidance_scale      = model_config.guidance_scale,
                width               = model_config.W,
                height              = model_config.H,
                video_length        = model_config.L,
            ).videos

            for idx, sample in enumerate(results):
                if args.run_csv or args.run_txt or default:
                    new_filename = f"{batch_videoids[idx]}.mp4"
                if args.run_json:
                    new_filename = f"{batch_videoids[idx]}-{batch_senids[idx]}.mp4"

                save_videos_grid(sample.unsqueeze(0), f"{savedir}/{new_filename}")
                print(f"save to {savedir}/{new_filename}")

    OmegaConf.save(model_config, f"{savedir}/model_config.yaml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--without-xformers", action="store_true")
    parser.add_argument("--human", action="store_true", help="Enable human mode for interactive video generation")
    parser.add_argument("--run-csv", type=str, default=None)
    parser.add_argument("--run-json", type=str, default=None)
    parser.add_argument("--run-txt", type=str, default=None)
    parser.add_argument("--save-path", type=str, default="outputs")
    parser.add_argument("--batch-size", type=int, default=1)
    
    args = parser.parse_args()
    main(args)
