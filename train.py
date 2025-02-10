# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
import math
import wandb
import logging
from pathlib import Path
import safetensors.torch
from copy import deepcopy
from tqdm.auto import tqdm
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, get_model_state_dict, set_model_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint

from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import T5Tokenizer, T5EncoderModel
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from diffusers.models.attention_processor import Attention

from utils.utils import get_optimizer, prepare_rotary_positional_embeddings, print_memory, reset_memory, encode_prompt, get_args

logger = get_logger(__name__)

def setup(rank, world_size, local_rank):
    # initialize the process group
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    # used for async save fsdp models
    save_group = dist.new_group(ranks=list(range(dist.get_world_size())), backend="gloo")
    return save_group

def cleanup():
    dist.destroy_process_group()

def get_custom_wrap_policy(modules_to_wrap):
    def custom_wrap_policy(module, recurse, unwrapped_params=None, nonwrapped_numel=None):
        return recurse or isinstance(module, modules_to_wrap)

    return custom_wrap_policy


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        if self.optimizer is not None:
            model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
            return {
                "model": model_state_dict,
                "optim": optimizer_state_dict
            }
        else:
            # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
            model_state_dict = get_model_state_dict(self.model)
            return {
                "model": model_state_dict,
            }

    def load_state_dict(self, state_dict):
        if self.optimizer is not None:
            # sets our state dicts on the model and optimizer, now that we've loaded
            set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optim"]
            )
        else:
            set_model_state_dict(
                self.model,
                model_state_dict=state_dict["model"],
            )


import pickle 
def save_pkl(file_name, obj):
    with open(file_name, 'wb') as fobj:
        pickle.dump(obj, fobj)

def load_pkl(file_name):
    with open(file_name, 'rb') as fobj:
        return pickle.load(fobj)

def loss_logging(loss, loss_name, rank, world_size, gradient_accumulation_steps, global_step, writer):
    loss_val = torch.tensor(loss.item()).to(torch.cuda.current_device())
    dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
    loss_val_avg = loss_val / world_size * gradient_accumulation_steps
    if rank == 0:
        assert writer is not None
        writer.add_scalar(loss_name, loss_val_avg, global_step)
    return loss_val_avg.item()

def get_model_paths(ckpt_path):
    model_names = {'model': 'model', 'model_ema': 'model_ema', 'global_step': 'global_step.pkl'}
    model_paths = {k: os.path.join(ckpt_path, v) for k, v in model_names.items()}
    return model_paths

def save_model(model, model_ema, optimizer, output_dir, save_group, global_step, rank, output_message):

    def _save(_model, _optimizer, _path):
        _state_dict = { "app": AppState(_model, _optimizer) }
        _checkpoint_future = dcp.async_save(_state_dict, checkpoint_id=_path, process_group=save_group)
        if _checkpoint_future is not None:
            _checkpoint_future.result()

    ckpt_path = os.path.join(output_dir, f"checkpoint-{global_step}")
    model_paths = get_model_paths(ckpt_path)
    _save(model, optimizer, model_paths['model'])
    if model_ema:
        _save(model_ema, None, model_paths['model_ema'])

    dist.barrier()

    if model_ema:
        # use a barrier to make sure training is done on all ranks
        transformer_checkpoint = model_ema.state_dict()
        output_name = "transformer_ema_step_{}.safetensor".format(str(global_step))
    else:
        transformer_checkpoint = model.state_dict()
        output_name = "transformer_step_{}.safetensor".format(str(global_step))

    if rank == 0:
        save_pkl(model_paths['global_step'], global_step)
        safetensors.torch.save_file(transformer_checkpoint, os.path.join(ckpt_path, output_name))
        ckpt_latest_path = os.path.abspath(os.path.join(output_dir, "checkpoint-latest"))
        if os.path.exists(ckpt_latest_path):
            os.unlink(ckpt_latest_path)
        os.symlink(ckpt_path, ckpt_latest_path)
    print(output_message)

def load_model(model, model_ema, optimizer, output_dir, resume_from_checkpoint):

    def _load(_model, _optimizer, _path):
        _state_dict = { "app": AppState(_model, _optimizer)}
        dcp.load(
            state_dict=_state_dict,
            checkpoint_id=_path,
        )
        print(f'Model loaded from path: {_path}')

    if not resume_from_checkpoint:
        return None

    ckpt_path = os.path.join(output_dir, f'checkpoint-{resume_from_checkpoint}')
    model_paths = get_model_paths(ckpt_path)

    if not os.path.exists(model_paths['model']) or \
        (model_ema is not None) and not os.path.exists(model_paths['model_ema']) or \
        not os.path.exists(model_paths['global_step']):
        print(
            f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        return None

    print(f'Resuming from checkpoint {ckpt_path}')

    _load(model, optimizer, model_paths['model'])
    if model_ema:
        _load(model_ema, None, model_paths['model_ema'])

    global_step = load_pkl(model_paths['global_step'])
    return global_step

class CollateFunction:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in data[0]]

        if self.load_tensors:
            prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)

        images = [x["image"] for x in data[0]]
        images = torch.stack(images).to(dtype=self.weight_dtype, non_blocking=True)

        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)

        return {
            "images": images,
            "videos": videos,
            "prompts": prompts,
        }

def main(args):
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    save_group = setup(rank, world_size, local_rank)

    # 1. setting up logging configuration    
    if rank == 0:
        wandb.init(project=args.wandb_project, name=args.wandb_name, notes=args.wandb_notes, sync_tensorboard=True)
        wandb.config.update(args)

    logging_dir = Path(args.output_dir, args.logging_dir)
    writer = None
    if logging_dir and rank == 0:
        writer = SummaryWriter(logging_dir)
        print(f"[RANK:{rank}] Tensorboard writer loaded.")

    # 4. stdout logging config
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # 4. stdout logging config
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # 5. training settings
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if rank==0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Prepare models and scheduler
    # Load T5
    tokenizer_1 = T5Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    
    text_encoder_1 = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 # use just bf16 for cogvideo5B

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
    )
    transformer._set_gradient_checkpointing = True

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    text_encoder_1.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(True)

    VAE_SCALING_FACTOR = vae.config.scaling_factor
    VAE_SCALE_FACTOR_SPATIAL = 2 ** (len(vae.config.block_out_channels) - 1)
    RoPE_BASE_HEIGHT = transformer.config.sample_height * VAE_SCALE_FACTOR_SPATIAL
    RoPE_BASE_WIDTH = transformer.config.sample_width * VAE_SCALE_FACTOR_SPATIAL

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.bfloat16

    if args.ema:
        transformer_ema = deepcopy(transformer)
    else:
        transformer_ema = None

    bfSixteen = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )
        
    device_mesh = init_device_mesh("cuda", (world_size,))
    wrap_policy = get_custom_wrap_policy((
        # CogVideoXBlock,
        # Attention,
        torch.nn.Linear
    ))
        
    transformer = FSDP(
        transformer,
        mixed_precision=bfSixteen,
        # auto_wrap_policy=wrap_policy, # default transformer auto-wrap policy
        auto_wrap_policy=wrap_policy,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=True),
        limit_all_gathers=True,
        device_mesh=device_mesh,
    )

    if transformer_ema is not None:
        transformer_ema = FSDP(
            transformer_ema,
            mixed_precision=bfSixteen,
            auto_wrap_policy=wrap_policy, # default transformer auto-wrap policy
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=True),
            limit_all_gathers=True,
            device_mesh=device_mesh,
        )

    # adjust the VAE/Text encoder
    vae = vae.to(torch.cuda.current_device(), dtype=weight_dtype)
    text_encoder_1 = text_encoder_1.to(torch.cuda.current_device(), dtype=weight_dtype)
    # FSDP don't allow move to cuda
    transformer = transformer.to(dtype=weight_dtype)

    # set trainable parameter
    trainable_modules = ["."]
    for name, param in transformer.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = False
                break

    num_layers = transformer.config.num_layers
    freeze_start = num_layers // 2
    for layer_idx, (name, param) in enumerate(transformer.named_parameters()):
       if 'transformer_blocks' in name:
            parts = name.split('.')
            block_idx = int(parts[2])
            if block_idx >= freeze_start:
                print(name, layer_idx, "False")
                param.requires_grad = False
            else:
                print(name, layer_idx, "True")
                param.requires_grad = True
        # if layer_idx >= freeze_start:
        #     print(name, layer_idx, "False")
        #     param.requires_grad = False
        # else:
        #     print(name, layer_idx, "True")
        #     param.requires_grad = True

    if args.scale_lr:
        global_batch_size = args.train_batch_size * world_size
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * global_batch_size
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    transformer_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer_parameters,
        "lr": args.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    optimizer = get_optimizer(
        params_to_optimize=params_to_optimize,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        beta3=args.beta3,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        prodigy_decouple=args.prodigy_decouple,
        prodigy_use_bias_correction=args.prodigy_use_bias_correction,
        prodigy_safeguard_warmup=args.prodigy_safeguard_warmup,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        use_torchao=args.use_torchao,
        use_deepspeed=False,
        use_cpu_offload_optimizer=args.use_cpu_offload_optimizer,
        offload_gradients=args.offload_gradients,
    )

    ################################# Test ######################################
    # Dataset and DataLoader
    from utils.dataloader import VideoDataset
    train_dataset = VideoDataset(
        csv_path=args.csv_path, 
        video_folder=args.video_folder,
        height=480,
        width=720,
        fps=8,
        max_num_frames=49,
        skip_frames_start=0,
        skip_frames_end=0,
    )
    
    sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size,)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=2 if args.dataloader_num_workers != 0 else None,
        persistent_workers=True if args.dataloader_num_workers != 0 else False,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

     # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if rank==0:
        tracker_name = args.tracker_name or "cogvideox-sft"
        print("===== Memory before training =====")
        reset_memory(torch.cuda.current_device())
        print_memory(torch.cuda.current_device())

    # Train!
    total_batch_size = args.train_batch_size * world_size * args.gradient_accumulation_steps

    if rank == 0:
        print("***** Running training *****")
        print(f"  Num trainable parameters = {num_trainable_parameters}")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num batches each epoch = {len(train_dataloader)}")
        print(f"  Num epochs = {args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    global_step = load_model(transformer, transformer_ema, optimizer, args.output_dir, args.resume_from_checkpoint)
    if global_step is None:
        args.resume_from_checkpoint = None
        global_step = 0
    initial_global_step = global_step
    first_epoch = global_step // num_update_steps_per_epoch

    # handles the case when saved checkpoint does not contain initial_lr
    for param_group in optimizer.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']  # Set to the current LR

    if args.use_cpu_offload_optimizer:
        # lr_scheduler = None
        # print(
        #     "CPU Offload Optimizer cannot be used with DeepSpeed or builtin PyTorch LR Schedulers. If "
        #     "you are training with those settings, they will be ignored."
        # )
        print("use_cpu_offload_optimizer for lr_scheduler not implemented for now!")
        return
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * world_size,
            num_training_steps=args.max_train_steps * world_size,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
            last_epoch = -1 if initial_global_step == 0 else initial_global_step
        )
    if rank==0:
        progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=(rank != 0),
        )

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    if args.load_tensors:
        del vae, text_encoder_1
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.cuda.current_device())

    alphas_cumprod = scheduler.alphas_cumprod.to(torch.cuda.current_device(), dtype=torch.float32)

    for epoch in range(first_epoch, args.num_train_epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            transformer.train()
            vae.requires_grad_(False)
            text_encoder_1.requires_grad_(False)

            videos = batch["videos"].to(torch.cuda.current_device(), non_blocking=True, dtype=weight_dtype)
            prompts = batch["prompts"]   

            # Encode videos
            with torch.no_grad():
                videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                latent_dist = vae.encode(videos).latent_dist

                video_latents = latent_dist.sample() * VAE_SCALING_FACTOR
                video_latents = video_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                # Encode prompts
                prompt_embeds_1 = encode_prompt(
                    tokenizer=tokenizer_1,
                    text_encoder=text_encoder_1,
                    prompt=prompts,
                    max_sequence_length=model_config.max_text_seq_length,
                    device=torch.cuda.current_device(),
                    dtype=weight_dtype,
                )

            # Sample noise that will be added to the latents
            patch_size_t = transformer.config.patch_size_t
            if patch_size_t is not None:
                ncopy = video_latents.shape[1] % patch_size_t
                # Copy the first frame ncopy times to match patch_size_t
                first_frame = video_latents[:, :1, :, :, :]  # Get first frame [B, 1, C, H, W]
                video_latents = torch.cat([first_frame.repeat(1, ncopy, 1, 1, 1), video_latents], dim=1)
                assert video_latents.shape[1] % patch_size_t == 0

            noise = torch.randn_like(video_latents)
            batch_size, num_frames, num_channels, height, width = video_latents.shape

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (batch_size,),
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )

            # Prepare rotary embeds
            image_rotary_emb = (
                prepare_rotary_positional_embeddings(
                    height=height * VAE_SCALE_FACTOR_SPATIAL,
                    width=width * VAE_SCALE_FACTOR_SPATIAL,
                    num_frames=num_frames,
                    vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
                    patch_size=model_config.patch_size,
                    patch_size_t=model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None,
                    attention_head_dim=model_config.attention_head_dim,
                    device=torch.cuda.current_device(),
                    base_height=RoPE_BASE_HEIGHT,
                    base_width=RoPE_BASE_WIDTH,
                )
                if model_config.use_rotary_positional_embeddings
                else None
            )

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = scheduler.add_noise(video_latents, noise, timesteps)
            model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None,
            ofs_embed_dim = model_config.ofs_embed_dim if hasattr(model_config, "ofs_embed_dim") else None,
            ofs_emb = None if ofs_embed_dim is None else noisy_model_input.new_full((1,), fill_value=2.0)

            # Predict the noise residual
            model_output = transformer(
                hidden_states=noisy_model_input,
                encoder_hidden_states=prompt_embeds_1,
                timestep=timesteps,
                ofs=ofs_emb,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]

            # print("===== After forward =====")
            # reset_memory(torch.cuda.current_device())
            # print_memory(torch.cuda.current_device())

            model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)

            weights = 1 / (1 - alphas_cumprod[timesteps])
            while len(weights.shape) < len(model_pred.shape):
                weights = weights.unsqueeze(-1)

            target = video_latents

            loss = torch.mean(
                (weights * (model_pred - target) ** 2).reshape(batch_size, -1),
                dim=1,
            )
            loss = loss.mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()      
            lr_scheduler.step()

            # print("===== After optimize =====")
            # reset_memory(torch.cuda.current_device())
            # print_memory(torch.cuda.current_device())

            # update ema:
            if (
                transformer_ema and
                global_step != 0 and 
                global_step % (args.ema_interval * args.gradient_accumulation_steps) == 0
            ):
                for tgt, src in zip(transformer_ema.parameters(), transformer.parameters()):
                    tgt.data.lerp_(src.data.to(tgt.device), 1 - args.ema_decay)

            if rank==0:
                progress_bar.update(1)

            global_step += 1

            model_output = None
            model_pred = None
            noisy_model_input = None
            prompt_embeds_1 = None
            prompt_embeds_2 = None
            prompt_attention_mask_2 = None
            timesteps = None
            ofs_emb = None
            image_rotary_emb = None
            del model_pred
            del model_output
            del noisy_model_input
            del prompt_embeds_1
            del prompt_embeds_2
            del prompt_attention_mask_2
            del timesteps
            del ofs_emb
            del image_rotary_emb
            free_memory()

            # Save the FSDP checkpoint for resuming (only in non-ema mode for training)
            if global_step % args.checkpointing_steps == 0:
                save_model(transformer, transformer_ema, optimizer, args.output_dir, save_group, global_step, rank, 'finished saving transformer')

            #logging
            last_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else args.learning_rate
            loss_val = loss_logging(loss, "loss", rank, world_size, args.gradient_accumulation_steps, global_step, writer)

            if rank==0:
                writer.add_scalar("lr", last_lr, global_step)
            
            logs = {"loss_val": loss_val, "lr": last_lr}
            if rank==0:
                progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
            
    dist.barrier()
    if rank == 0:
        wandb.finish()
    cleanup()


if __name__ == "__main__":
    args = get_args()
    try:
        main(args)
    except:
        raise Exception()