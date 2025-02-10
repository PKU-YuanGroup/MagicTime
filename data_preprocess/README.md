# <u>Data Preprocessing Pipeline</u> by *MagicTime*
This repo describes how to process your own data like [ChronoMagic](https://huggingface.co/datasets/BestWishYsh/ChronoMagic) datasets in the [MagicTime](https://arxiv.org/abs/2404.05014) paper.

## 🗝️ Usage

```bash
#!/bin/bash

# Global variables
INPUT_FOLDER="./step_0"
OUTPUT_FOLDER_STEP_1="./step_1"
API_KEY="XXX"
NUM_WORKERS=8

# File paths
FRAME_CAPTION_FILE="./2_1_gpt_frames_caption.json"
GROUP_FRAMES_FILE="./2_1_temp_group_frames.json"
UPDATED_FRAME_CAPTION_FILE="./2_2_updated_gpt_frames_caption.json"
UNMATCHED_FRAME_CAPTION_FILE="./2_2_temp_unmatched_gpt_frames_caption.json"
UNORDERED_FRAME_CAPTION_FILE="./2_2_temp_unordered_gpt_frames_caption.json"
FINAL_USEFUL_FRAME_CAPTION_FILE="./2_2_final_useful_gpt_frames_caption.json"
VIDEO_CAPTION_FILE="./3_1_gpt_video_caption.json"
UNMATCHED_VIDEO_CAPTION_FILE="./3_2_temp_unmatched_gpt_video_caption.json"
EXCLUDE_BY_FRAME_VIDEO_CAPTION_FILE="./3_2_temp_exclude_by_frame_gpt_video_caption.json"
FINAL_USEFUL_VIDEO_CAPTION_FILE="./3_2_final_useful_gpt_video_caption.json"
FINAL_CSV_FILE="./all_clean_data.csv"

# Step 1: Extract and resize frames
python step0_extract_frame_resize.py --input_folder "$INPUT_FOLDER" --output_folder "$OUTPUT_FOLDER_STEP_1"

# Step 2.1: Generate frame captions using GPT-4V
python step2_1_GPT4V_frame_caption.py --api_key "$API_KEY" --num_workers "$NUM_WORKERS" \
    --output_file "$FRAME_CAPTION_FILE" --group_frames_file "$GROUP_FRAMES_FILE" --image_directories "$OUTPUT_FOLDER_STEP_1"

# Step 2.2: Preprocess frame captions
python step2_2_preprocess_frame_caption.py --file_path "$FRAME_CAPTION_FILE" \
    --updated_file_path "$UPDATED_FRAME_CAPTION_FILE" --unmatched_file_path "$UNMATCHED_FRAME_CAPTION_FILE" \
    --unordered_file_path "$UNORDERED_FRAME_CAPTION_FILE" --final_useful_data_file_path "$FINAL_USEFUL_FRAME_CAPTION_FILE"

# Step 3.1: Generate concise video captions using GPT-4V
python step3_1_GPT4V_video_caption_concise.py --num_workers "$NUM_WORKERS" \
    --input_file "$FINAL_USEFUL_FRAME_CAPTION_FILE" --output_file "$VIDEO_CAPTION_FILE"

# Optional: Generate detailed video captions (uncomment to enable)
# python step3_1_GPT4V_video_caption_detail.py --num_workers "$NUM_WORKERS" \
#     --input_file "$FINAL_USEFUL_FRAME_CAPTION_FILE" --output_file "$VIDEO_CAPTION_FILE"

# Step 3.2: Preprocess video captions
python step3_2_preprocess_video_caption.py --file_path "$VIDEO_CAPTION_FILE" \
    --updated_file_path "$VIDEO_CAPTION_FILE" --unmatched_data_path "$UNMATCHED_VIDEO_CAPTION_FILE" \
    --exclude_by_frame_data_path "$EXCLUDE_BY_FRAME_VIDEO_CAPTION_FILE" --final_useful_data_path "$FINAL_USEFUL_VIDEO_CAPTION_FILE"

# Step 4: Create the final dataset in WebVid format
python step4_1_create_webvid_format.py --caption_file_path "$FINAL_USEFUL_VIDEO_CAPTION_FILE" \
    --output_csv_file_path "$FINAL_CSV_FILE"
```
