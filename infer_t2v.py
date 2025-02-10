import os
import time
import argparse
from safetensors.torch import load_file

import torch
from diffusers.utils import export_to_video
from diffusers import CogVideoXPipeline, CogVideoXTransformer3DModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, set_seed

def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

def rephrase_prompt(processor, model, short_prompt=" ", seed=42):
    rephraser_prompt = "Based on a video and a short phrase: {}, expand it to a medium length prompt for video generation model. \
            Make sure to start the prompt with the content of short phrase provided, and central around this action. If there's no short phrase, build the prompt \
            with plausible motion descriptions based on the image but do make sure the motion description is not too intense, use common and subtle action (like \
            person wave hand, smile, blinks eyes, happily smile, gaze through the camera, and etc).  Adjust based on the short prompt to not use intense action like `running, fighting, walking.` \
            Make sure do not use still actions with no motions like `stand still, stand, stay,`, always make sure the the prompt contains an active action but not too intense. \
            Make the prompt is precise without hallucination and do not make any conjecture, and avoid use any unsafe, bad languages. Make sure the action is simple and common daily life activities, \
            and include only one or two major actions. A good example of prompt is `a young woman blinks her eyes, turns her head to the side as she gazes thoughtfully \
            into the camera with her beautiful and blinking eyes,  blinking. Her head subtly turns towards the viewer, adding a sense of motion to the scene. \
            The natural light casts soft shadows on the background, enhancing the mood and emphasizing her.` Make sure to describe some emotion and pose motion for the person `smile, laungh, tilts heads, blinks and etc` \
            Another example is `A person stands while smiling confidently in a space suit adorned  with various inscriptions and patches. He smiles and holds a helmet in their left hand while waving and greating, \
            their face illuminated by a warm, friendly smile. The person's gaze is directed towards the camera, inviting viewers to share in their excitement and dreams of space exploration.` \
            Avoid description about backgrounds or other objects. Make the prompt succinct. The short phrasem for this video is: {}".format(short_prompt, short_prompt)

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": rephraser_prompt}
    ]

    with torch.no_grad():
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = processor([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return output_text

# Load the CogVideo T2V pipeline 
def load_cogvideo_t2v(model_path="THUDM/CogVideoX-5b",
                      custom_transformer_state_dict=None,
                      vae_tile=True,
                      transformer_compile=False,
                      cpu_offload=False,
                      seq_cpu_offload=False,
                      type=torch.bfloat16, 
                      device="cuda"):

    transformer = CogVideoXTransformer3DModel.from_pretrained_consisid(
        model_path,
        subfolder="transformer",
        load_ckpt=True,
        transformer_additional_kwargs={
                        'in_channels': 16,
                        'torch_dtype': torch.bfloat16,
                        'is_timeaware_connector': True,
                        'is_flexible_max_length': True,
                        'is_mmdit': False,
                        'is_sp': False,
        }
    ).to(device, torch.bfloat16)
    pipe =  CogVideoXPipeline.from_pretrained(model_path, transformer=transformer).to(device, torch.bfloat16)

    if custom_transformer_state_dict is not None:
        if custom_transformer_state_dict.endswith(".safetensor"):
            # Load state dict from safetensors format
            state_dict = load_file(custom_transformer_state_dict)
        else:
            # Standard PyTorch checkpoint
            state_dict = torch.load(custom_transformer_state_dict, map_location=device)
    
        # load the checkpoint
        load_info = pipe.transformer.load_state_dict(state_dict, strict=True)
        print("Loaded transformer checkpoint:", load_info)
    
    # compile and optimize the inference speed
    if transformer_compile:
        pipe.transformer.to(memory_format=torch.channels_last)
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)

    # memory offload optimization: Without enabling cpu offloading, memory usage is 33 GB; With enabling cpu offloading, memory usage is 19 GB
    if cpu_offload:
        pipe.enable_model_cpu_offload()

    # significantly reduce memory usage at the cost of slow inference; When enabled, memory usage is under 4 GB
    if seq_cpu_offload:
        pipe.enable_sequential_cpu_offload()

    # With enabling cpu offloading and tiling, memory usage is 11 GB
    if vae_tile:
        pipe.vae.enable_tiling()

    return pipe

# load Qwen as rephraser
def load_qwen(model_path="Qwen/Qwen2.5-VL-7B-Instruct", type=torch.bfloat16, device="cuda"):

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, 
    # especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=type,
        attn_implementation="flash_attention_2",   # need installation
        # device_map="auto",
    ).to(device)

    # default processer
    processor = AutoProcessor.from_pretrained(model_path)

    return processor, model

def main(args):
    # ====================================================== Run Inference =================================================
    seed = args.seed
    set_seed(seed)

    # CUDA Device
    device = "cuda"

    qwen_path = args.qwen_path
    model_path = args.model_path

    # DiT Checkpoint
    # transformer_ckpt = "transformer_step_1000.safetensor"
    transformer_ckpt = None

    prompt_file = args.prompt_file
    negative_prompt = "jittery, choppy, worst quality, unnatural, intense motion, abrupt motion, inconsistent object movement, shaky or unstable camera motion, unintentional artifacts, normal quality, low quality, low res, blurry, distortion, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, distortion"

    output_path = args.output_path
    txt_output_path = args.txt_output_path
    current_time = time.strftime("%Y%m%d_%H%M%S")
    create_folder_if_not_exist(os.path.join(output_path, current_time))
    os.makedirs(txt_output_path, exist_ok=True)

    # Load the T2V model and prompt rephraser
    print("Load Pipeline")
    t2v_pipeline = load_cogvideo_t2v(model_path, 
                                     custom_transformer_state_dict=transformer_ckpt, 
                                     device=device)

    print("Load Qwen")
    processor, rephraser  = load_qwen(model_path=qwen_path, 
                                      device=device)
    
    # Open the file in read mode
    with open(prompt_file, 'r') as file:
        lines = file.readlines()
        generator = torch.Generator(device="cuda").manual_seed(seed)

        for line in lines:
            # rephrase
            prompt = rephrase_prompt(
                                     processor=processor,
                                     model=rephraser,
                                     short_prompt=line,
                                     seed=seed,
                                    )

            print("Prompt is:", prompt, line)

            # # find the suitable resolution
            video = t2v_pipeline(
                        prompt=prompt[0], 
                        negative_prompt=negative_prompt,
                        use_dynamic_cfg=True, 
                        generator=generator,
                        num_frames=49,
                        height=480, 
                        width=720, 
                        guidance_scale=6,
                    )
            
            with open(os.path.join(txt_output_path, "{}.txt".format(current_time)), "a") as f:
                f.write("{}, {}\n".format(prompt[0]))


            video_output_path = os.path.join(output_path, current_time, "{}_{}.mp4".format(seed))

            export_to_video(video.frames[0], video_output_path, fps=args.fps)
            print(line, prompt, output_path, "done")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--qwen_path', type=str, default="/storage/ysh/Ckpts/Qwen2.5-7B-Instruct/")
    parser.add_argument('--model_path', type=str, default="/storage/ysh/Ckpts/OmniConsisID_544p")
    parser.add_argument('--output_path', type=str, default="samples/test")
    parser.add_argument('--prompt_file', type=str, default="inference/test_prompts.txt")
    parser.add_argument('--txt_output_path', type=str, default="samples/txt")
    parser.add_argument('--fps', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
