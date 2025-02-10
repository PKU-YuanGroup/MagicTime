import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from threading import Lock
from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed


txt_prompt = '''
Imagine you're an expert data annotator with a specialization in summarizing time-lapse videos. You will be supplied with "Video_Reasoning", "8_Key-Frames_Reasoning", and "8_Key-Frames_Captioning" from a video, your task is to craft a concise summary for the given time-lapse video.

Since only textual information is given, you can employ logical deductions to bridge any informational gaps if necessary. For guidance on the expected output format and content length (no more than 70 words), refer to the provided examples:

"Video_Summary": Time-lapse of a ciplukan fruit growing from a small bud to a mature, rounded form among leaves, gradually enlarging and smoothing out by the video's end.

"Video_Summary": Time-lapse of red onion bulbs sprouting and growing over 10 days: starting dormant, developing shoots and roots by Day 2, significant growth by Day 6, and full development by Day 10.

"Video_Summary": "{Video Summary}"

Attention: Do not reply outside the example template! The process of reasoning and thinking should not be included in the {Video Summary}! Do not use words similar to by frame or at frame! Below are the Video, Video_Reasoning, Frame_Reasoning and Frame_Captioning.
'''

# Global lock for thread-safe file operations
file_lock = Lock()

# Function to create prompts for the GPT-4 Vision API
def create_prompts(txt_prompt, data):
    prompts = {}
    for video_id, value in tqdm(data.items(), desc="Creating prompts"):
        prompt = [{"type": "text", "text": txt_prompt}]
        prompt.append({"type": "text", "text": f'''The "Video_Reasoning" is: {value['Video_Reasoning']}'''})
        prompt.append({"type": "text", "text": f'''The "8_Key-Frames_Reasoning" are: {value['Frame_Reasoning']}'''})
        prompt.append({"type": "text", "text": f'''The "8_Key-Frames_Captioning" are: {value['Frame_Captioning']}'''})
        prompts[video_id] = prompt
    return prompts

def has_been_processed(video_id, output_file):
    with file_lock:
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                data = json.load(f)
                if video_id in data:
                    print(f"Video ID {video_id} has already been processed.")
                    return True
        return False

def load_existing_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            print(f"Loading existing results from {file_path}")
            return json.load(file)
    else:
        print(f"No existing results file found at {file_path}. Creating a new file.")
        with open(file_path, 'w') as file:
            empty_data = {}
            json.dump(empty_data, file)
            return empty_data

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(100))
def call_gpt(prompt, model_name="gpt-4-vision-preview", api_key=None):
    client = OpenAI(api_key=api_key)
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=1024,
    )
    return chat_completion.choices[0].message.content

def save_output(video_id, prompt, output_file, api_key):
    if not has_been_processed(video_id, output_file):
        result = call_gpt(prompt, api_key=api_key)
        with file_lock:
            with open(output_file, 'r+') as f:
                # Read the current data and update it
                data = json.load(f)
                data[video_id] = result
                f.seek(0)  # Rewind file to the beginning
                json.dump(data, f, indent=4)
                f.truncate()  # Truncate file to new size
        print(f"Processed and saved output for Video ID {video_id}")

def main(num_workers, all_prompts, output_file, api_key):
    # Load existing results
    existing_results = load_existing_results(output_file)

    # Filter prompts for video IDs that have not been processed
    unprocessed_prompts = {vid: prompt for vid, prompt in all_prompts.items() if vid not in existing_results}
    if not unprocessed_prompts:
        print("No unprocessed video IDs found. All prompts have already been processed.")
        return

    print(f"Processing {len(unprocessed_prompts)} unprocessed video IDs.")

    progress_bar = tqdm(total=len(unprocessed_prompts))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(save_output, video_id, prompt, output_file, api_key): video_id 
            for video_id, prompt in unprocessed_prompts.items()
        }

        for future in as_completed(future_to_index):
            progress_bar.update(1)
            try:
                future.result()
            except Exception as e:
                print(f"Error processing video ID {future_to_index[future]}: {e}")

    progress_bar.close()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate video captions using GPT4V.")
    parser.add_argument("--api_key", type=int, default=None, help="OpenAI API key.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker threads for processing.")
    parser.add_argument("--input_file", type=str, default="./2_2_final_useful_gpt_frames_caption.json", help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, default="./3_1_gpt_video_caption.json", help="Path to save the generated video captions.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Load data from the input file
    with open(args.input_file, 'r') as file:
        data = json.load(file)

    # Generate prompts
    prompts = create_prompts(txt_prompt, data)

    # Execute main processing function
    main(args.num_workers, prompts, args.output_file, args.api_key)