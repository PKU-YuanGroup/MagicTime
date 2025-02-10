import gc
import os
import imageio
import inspect
import argparse
import numpy as np
from einops import rearrange
from typing import List, Optional, Union, Tuple

import torch
import torchvision
from accelerate.logging import get_logger
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from transformers import T5EncoderModel, T5Tokenizer, Qwen2Tokenizer

logger = get_logger(__name__)

def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds

def encode_prompt(
    tokenizer,
    text_encoder,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    prompt: Optional[Union[str, List[str]]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    with torch.no_grad():
        prompt_embeds = _get_t5_prompt_embeds(
            tokenizer,
            text_encoder,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
            text_input_ids=text_input_ids,
        )
    return prompt_embeds

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w").to(torch.float32).to("cpu")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def get_optimizer(
    params_to_optimize,
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.95,
    beta3: float = 0.98,
    epsilon: float = 1e-8,
    weight_decay: float = 1e-4,
    prodigy_decouple: bool = False,
    prodigy_use_bias_correction: bool = False,
    prodigy_safeguard_warmup: bool = False,
    use_8bit: bool = False,
    use_4bit: bool = False,
    use_torchao: bool = False,
    use_deepspeed: bool = False,
    use_cpu_offload_optimizer: bool = False,
    offload_gradients: bool = False,
) -> torch.optim.Optimizer:
    optimizer_name = optimizer_name.lower()

    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=weight_decay,
        )

    if use_8bit and use_4bit:
        raise ValueError("Cannot set both `use_8bit` and `use_4bit` to True.")

    if (use_torchao and (use_8bit or use_4bit)) or use_cpu_offload_optimizer:
        try:
            import torchao

            torchao.__version__
        except ImportError:
            raise ImportError(
                "To use optimizers from torchao, please install the torchao library: `USE_CPP=0 pip install torchao`."
            )

    if not use_torchao and use_4bit:
        raise ValueError("4-bit Optimizers are only supported with torchao.")

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy", "came"]
    if optimizer_name not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {optimizer_name}. Supported optimizers include {supported_optimizers}. Defaulting to `AdamW`."
        )
        optimizer_name = "adamw"

    if (use_8bit or use_4bit) and optimizer_name not in ["adam", "adamw"]:
        raise ValueError("`use_8bit` and `use_4bit` can only be used with the Adam and AdamW optimizers.")

    if use_8bit:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if optimizer_name == "adamw":
        if use_torchao:
            from torchao.prototype.low_bit_optim import AdamW4bit, AdamW8bit

            optimizer_class = AdamW8bit if use_8bit else AdamW4bit if use_4bit else torch.optim.AdamW
        else:
            optimizer_class = bnb.optim.AdamW8bit if use_8bit else torch.optim.AdamW

        init_kwargs = {
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }

    elif optimizer_name == "adam":
        if use_torchao:
            from torchao.prototype.low_bit_optim import Adam4bit, Adam8bit

            optimizer_class = Adam8bit if use_8bit else Adam4bit if use_4bit else torch.optim.Adam
        else:
            optimizer_class = bnb.optim.Adam8bit if use_8bit else torch.optim.Adam

        init_kwargs = {
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }

    elif optimizer_name == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        init_kwargs = {
            "lr": learning_rate,
            "betas": (beta1, beta2),
            "beta3": beta3,
            "eps": epsilon,
            "weight_decay": weight_decay,
            "decouple": prodigy_decouple,
            "use_bias_correction": prodigy_use_bias_correction,
            "safeguard_warmup": prodigy_safeguard_warmup,
        }

    elif optimizer_name == "came":
        try:
            import came_pytorch
        except ImportError:
            raise ImportError("To use CAME, please install the came-pytorch library: `pip install came-pytorch`")

        optimizer_class = came_pytorch.CAME

        init_kwargs = {
            "lr": learning_rate,
            "eps": (1e-30, 1e-16),
            "betas": (beta1, beta2, beta3),
            "weight_decay": weight_decay,
        }

    if use_cpu_offload_optimizer:
        from torchao.prototype.low_bit_optim import CPUOffloadOptimizer

        if "fused" in inspect.signature(optimizer_class.__init__).parameters:
            init_kwargs.update({"fused": True})

        optimizer = CPUOffloadOptimizer(
            params_to_optimize, optimizer_class=optimizer_class, offload_gradients=offload_gradients, **init_kwargs
        )
    else:
        optimizer = optimizer_class(params_to_optimize, **init_kwargs)

    return optimizer


def get_gradient_norm(parameters):
    norm = 0
    for param in parameters:
        if param.grad is None:
            continue
        local_norm = param.grad.detach().data.norm(2)
        norm += local_norm.item() ** 2
    norm = norm**0.5
    return norm


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    patch_size_t: int = None,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    if patch_size_t is None:
        # CogVideoX 1.0
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )
    else:
        # CogVideoX 1.5
        base_num_frames = (num_frames + patch_size_t - 1) // patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(base_size_height, base_size_width),
        )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def reset_memory(device: Union[str, torch.device]) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)


def print_memory(device: Union[str, torch.device]) -> None:
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(f"{memory_allocated=:.3f} GB")
    print(f"{max_memory_allocated=:.3f} GB")
    print(f"{max_memory_reserved=:.3f} GB")

def _get_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--custom_transformer_state_dict",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained DiT checkpoint.",
    )    
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )


def _get_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help=("YAML file for V0 dataset"),
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        help=("Path to a CSV file if loading prompts/video paths using this format."),
    )
    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        help="The column of the dataset containing videos. Or, the name of the file in `--data_root` folder containing the line-separated path to video data.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing the instance prompt for each video. Or, the name of the file in `--data_root` folder containing the line-separated instance prompts.",
    )
    parser.add_argument(
        "--id_token",
        type=str,
        default=None,
        help="Identifier token appended to the start of each prompt if provided.",
    )
    parser.add_argument(
        "--load_tensors",
        action="store_true",
        help="Whether to use a pre-encoded tensor dataset of latents and prompt embeddings instead of videos and text prompts. The expected format is that saved by running the `prepare_dataset.py` script.",
    )
    parser.add_argument(
        "--random_flip",
        type=float,
        default=None,
        help="If random horizontal flip augmentation is to be used, this should be the flip probability.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Whether or not to use the pinned memory setting in pytorch dataloader.",
    )


def _get_validation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="One or more prompt(s) that is used during validation to verify that the model is learning. Multiple validation prompts should be separated by the '--validation_prompt_seperator' string.",
    )
    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        help="One or more image path(s)/URLs that is used during validation to verify that the model is learning. Multiple validation paths should be separated by the '--validation_prompt_seperator' string. These should correspond to the order of the validation prompts.",
    )
    parser.add_argument(
        "--validation_prompt_separator",
        type=str,
        default=":::",
        help="String that separates multiple validation prompts",
    )
    parser.add_argument(
        "--num_validation_videos",
        type=int,
        default=1,
        help="Number of videos that should be generated during validation per `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help="Run validation every X training steps. Validation consists of running the validation prompt `args.num_validation_videos` times.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6,
        help="The guidance scale to use while sampling validation videos.",
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=False,
        help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.",
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        default=False,
        help="Whether or not to enable model-wise CPU offloading when performing validation/testing to save memory.",
    )


def _get_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--rank", type=int, default=64, help="The rank for LoRA matrices.")
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="The lora_alpha to compute scaling factor (lora_alpha / rank) for LoRA matrices.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU. "
            "Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this "
            "argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-sft",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="All input videos are resized to this width.",
    )
    parser.add_argument(
        "--video_reshape_mode",
        type=str,
        default=None,
        help="All input videos are reshaped to this mode. Choose between ['center', 'random', 'none']",
    )
    parser.add_argument("--fps", type=int, default=8, help="All input videos will be used at this FPS.")
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=49,
        help="All input videos will be truncated to these many frames.",
    )
    parser.add_argument(
        "--skip_frames_start",
        type=int,
        default=0,
        help="Number of frames to skip from the beginning of each input video. Useful if training data contains intro sequences.",
    )
    parser.add_argument(
        "--skip_frames_end",
        type=int,
        default=0,
        help="Number of frames to skip from the end of each input video. Useful if training data contains outro sequences.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--disc_resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--ema_resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )
    parser.add_argument(
        "--noised_image_dropout",
        type=float,
        default=0.05,
        help="Image condition dropout probability when finetuning image-to-video.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="ema decay rate",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="use ema"
    )
    parser.add_argument(
        "--ema_interval",
        type=int,
        default=10,
        help="ema update step interval"
    )
    parser.add_argument(
        "--dit_cp_size",
        type=int,
        default=1,
        help="context parallel size for DiT"
    )

def _get_optimizer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy", "came"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Whether or not to use 8-bit optimizers from `bitsandbytes` or `bitsandbytes`.",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Whether or not to use 4-bit optimizers from `torchao`.",
    )
    parser.add_argument(
        "--use_torchao", action="store_true", help="Whether or not to use the `torchao` backend for optimizers."
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument(
        "--prodigy_decouple",
        action="store_true",
        help="Use AdamW style decoupled weight decay.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for optimizer.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--prodigy_use_bias_correction",
        action="store_true",
        help="Turn on Adam's bias correction.",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )
    parser.add_argument(
        "--use_cpu_offload_optimizer",
        action="store_true",
        help="Whether or not to use the CPUOffloadOptimizer from TorchAO to perform optimization step and maintain parameters on the CPU.",
    )
    parser.add_argument(
        "--offload_gradients",
        action="store_true",
        help="Whether or not to offload the gradients to CPU when using the CPUOffloadOptimizer from TorchAO.",
    )
    parser.add_argument('--local-rank',
                        type=int, 
                        default=0, 
                        help='Local rank. Necessary for using the torch.distributed.launch utility.')



def _get_configuration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="jacob-cogvideo-diffusers-i2v",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="testrun",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--wandb_notes",
        type=str,
        default="",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--nccl_timeout",
        type=int,
        default=600,
        help="Maximum timeout duration before which allgather, or related, operations fail in multi-GPU/multi-node training settings.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

def _get_distillation_args(parser):
    parser.add_argument(
        "--num_ddim_timesteps",
        type=int,
        default=100,
        help="Number of timesteps for DDIM sampling for teacher model",
    )
    parser.add_argument(
        "--num_inference_steps_min",
        type=int,
        default=100,
        help="min number of inference steps for the distilled student model",
    )
    parser.add_argument(
        "--num_inference_steps_max",
        type=int,
        default=100,
        help="max number of inference steps for the distilled student model",
    )
    parser.add_argument(
        "--w_min",
        type=float,
        default=5.,
        help="min value of cfg",
    )
    parser.add_argument(
        "--w_max",
        type=float,
        default=7.,
        help="max value of cfg",
    )
    parser.add_argument(
        "--s_ratio",
        type=float,
        default=0.3,
        help="max value of cfg",
    )

    parser.add_argument(
        "--log_inference_steps",
        type=int,
        default=25,
        help="number of inference steps to run",
    )
    parser.add_argument(
        "--log_guidance_scale",
        type=float,
        default=6,
        help="The guidance scale to use while sampling validation videos.",
    )
    parser.add_argument(
        "--adv_weight",
        type=float,
        default=1.0,
        help="The weight of the gan loss.",
    )

def _get_multisubject_args(parser):
    parser.add_argument(
        "--multimodal_llm_path",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Path to pretrained multimodal llm checkpoint.",
    )
    parser.add_argument(
        "--num_reference_tokens",
        type=int,
        default=226,
        help="number of reference tokens"
    )
    parser.add_argument(
        "--unfreeze_mllm",
        action="store_true",
        default=False,
        help="whether to unfreeze mllm",
    )
    parser.add_argument(
        "--pretrained_anyref_path",
        type=str,
        default=None,
        help="Path to pretrained anyref checkpoint.",
    )
    
def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")

    _get_model_args(parser)
    _get_dataset_args(parser)
    _get_training_args(parser)
    _get_validation_args(parser)
    _get_optimizer_args(parser)
    _get_configuration_args(parser)
    _get_distillation_args(parser)
    _get_multisubject_args(parser)

    return parser.parse_args()