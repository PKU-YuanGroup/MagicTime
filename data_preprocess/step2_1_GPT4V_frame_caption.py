import os
import re
import json
import base64
import argparse
from tqdm import tqdm
from openai import OpenAI
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_exponential, stop_after_attempt


txt_prompt = '''
Suppose you are a data annotator, specialized in generating captions for time-lapse videos. You will be supplied with eight key frames extracted from a video, each with a filename labeled with its position in the video sequence. Your task is to generate a caption for each frame, focusing on the primary subject and integrating all discernible elements. Note: These captions should be brief and concise, avoiding redundancy. 

Your analysis should demonstrate a deep understanding of real-world physics, encompassing aspects such as gravity and elasticity, and align with the principles of perspective geometry in photography. Ensure object identification consistency across all frames, even if an object is temporarily out of sight. Employ logical deductions to bridge any informational gaps. Begin each caption with a brief reasoning statement, showcasing your analytical approach. For guidance on the expected format, refer to the provided examples:

Brief Reasoning Statement: The images provided are sequential frames from a time-lapse video depicting the blooming stages of a yellow flower, likely a ranunculus. The sequence is forward, showing a natural progression from bud to full bloom. Time-related information is not included in these frames. I will describe each frame accordingly.
"[_2p6vHyth14]": {
"Reasoning": [
"Frame 0: This is the first frame, starting the sequence. The flower is in its initial stages, with petals tightly closed.",
"Frame 224: The petals appear slightly more open than in the first frame, indicating the progression of blooming.",
"Frame 448: The bloom has progressed further; petals are more open than in the previous frame, suggesting the continuation of the blooming process.",
"Frame 672: Continuity in the blooming process is evident, with petals unfurling more than in the last frame.",
"Frame 896: The flower is more open than in frame 672, indicating an advanced stage of the blooming process.",
"Frame 1120: The flower is nearing full bloom, with a majority of the petals open and the inner ones starting to loosen.",
"Frame 1344: The blooming process is almost complete, with the flower more open than in frame 1120 and the center more visible.",
"Frame 1570: This final frame likely represents the peak of the bloom, with the flower fully open and all petals relaxed."
],
"Captioning": [
"Frame 0: Closed yellow ranunculus bud amidst green foliage.",
"Frame 224: Yellow ranunculus bud beginning to open, with green sepals visible.",
"Frame 448: Opening yellow ranunculus with distinct petal layers.",
"Frame 672: Further unfurled yellow ranunculus, petals spreading outward.",
"Frame 896: Half-open yellow ranunculus, with inner petals still tightly clustered.",
"Frame 1120: Nearly fully bloomed yellow ranunculus, with central petals loosening.",
"Frame 1344: Yellow ranunculus in full bloom, center clearly visible amidst open petals.",
"Frame 1570: Fully bloomed yellow ranunculus with a fully visible center and relaxed petals."
]
}

Brief Reasoning Statement: The images show the germination and growth process of a plant, identified as spinach, over a span of 46 days. This time-lapse video captures the transformation from soil to a fully developed plant in a forward sequence. Time-related information is present, indicating the duration of the captured growth process. I will describe each frame accordingly.
"[pVmX1v1hDc]_0001": {
"Reasoning": [
"Frame 0: This is the initial stage where the soil is moist, likely right after sowing the seeds.",
"Frame 69: The soil surface shows signs of disturbance, possibly from seeds beginning to germinate.",
"Frame 138: Germination has occurred, evident from the emergence of seedlings breaking through the soil.",
"Frame 207: The seedlings have elongated and the first true leaves are beginning to form.",
"Frame 276: Growth is evident with larger true leaves, and the plant is entering the vegetative stage.",
"Frame 345: The plants are more developed with a denser leaf canopy, indicating healthy vegetative growth.",
"Frame 414: The spinach plants are fully developed with large leaves, ready for harvesting.",
"Frame 485: The plants are at full maturity with a thick canopy of leaves, showing the complete growth cycle."
],
"Captioning": [
"Frame 0: Moist soil on Day 1 after sowing spinach seeds.",
"Frame 69: Soil surface showing early signs of spinach seed germination on Day 6.",
"Frame 138: Spinach seedlings emerging from soil on Day 10.",
"Frame 207: Elongated spinach seedlings with first true leaves on Day 16.",
"Frame 276: Spinach showing significant leaf growth on Day 24.",
"Frame 345: Denser and larger spinach leaves visible on Day 31.",
"Frame 414: Mature spinach plants with large leaves ready for harvest on Day 39.",
"Frame 485: Thick canopy of mature spinach leaves on Day 46."
]
}

{Brief Reasoning Statement: Must include time-related information and description of forward processes}
"{Enter the prefix of the image to represent the id}": {
"Reasoning": [
" ",
" ",
" ",
" ",
" ",
" ",
" ",
" "
],
"Captioning": [
" ",
" ",
" ",
" ",
" ",
" ",
" ",
" "
]
}

Attention: Do not reply outside the example template! Below are the video title and input frames:
'''

# Global lock for thread-safe file operations
file_lock = Lock()

# Function to get all image filenames in the specified directory
def get_image_filenames(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in image_extensions]

# Function to parse the video ID from the image file name
def parse_video_id(filename):
    match = re.match(r'(.+)_frame\d+\.png', filename)
    return match.group(1) if match else None

# Function to convert image to base64
def image_b64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Function to group images
def group_images_by_video_id(filenames):
    images_by_video = {}
    for filename in tqdm(filenames, desc="Grouping images"):
        video_id = parse_video_id(filename)
        if video_id:
            if video_id not in images_by_video:
                images_by_video[video_id] = []
            images_by_video[video_id].append(filename)
    
    valid_groups = {video_id: images for video_id, images in images_by_video.items() if len(images) == 8}
    return valid_groups

# Function to create prompts for the GPT-4 Vision API
def create_prompts(grouped_images, image_directory, txt_prompt):
    prompts = {}
    for video_id, group in tqdm(grouped_images.items(), desc="Creating prompts"):
        # Initialize the prompt with the given text prompt
        prompt = [{"type": "text", "text": txt_prompt}]
        
        # Append information about each image in the group
        for image_name in group:
            image_path = os.path.join(image_directory, image_name.strip())
            b64_image = image_b64(image_path)
            prompt.append({"type": "text", "text": image_name.strip()})
            prompt.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}})
        
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

def extract_frame_number(filename):
    # Extract the number after 'frame' and convert to integer
    return int(filename.split('_frame')[-1].split('.')[0])

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
        max_tokens=2048,
    )
    print(chat_completion)
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
    parser = argparse.ArgumentParser(description="Process video frame captions.")
    parser.add_argument("--api_key", type=int, default=None, help="OpenAI API key.")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of worker threads for processing.")
    parser.add_argument("--output_file", type=str, default="./2_1_gpt_frames_caption.json", help="Path to the output JSON file.")
    parser.add_argument("--group_frames_file", type=str, default="./2_1_temp_group_frames.json", help="Path to save grouped frame metadata.")
    parser.add_argument("--image_directories", type=str, nargs="+", default=["./step_1"], help="List of directories containing images.")
    
    # Parse command-line arguments
    args = parser.parse_args()

    all_prompts = {}
    all_grouped_images = {}

    # Process each image directory
    for directory in args.image_directories:
        filenames = get_image_filenames(directory)
        grouped_images = group_images_by_video_id(filenames)
        
        # Sort images within each video group
        for video_id in grouped_images:
            grouped_images[video_id].sort(key=extract_frame_number)

        all_grouped_images.update(grouped_images)  # Merge into a single dictionary

        # Generate prompts
        prompts = create_prompts(grouped_images, directory, txt_prompt)
        all_prompts.update(prompts)

    # Save grouped images metadata
    with open(args.group_frames_file, 'w') as file:
        json.dump(all_grouped_images, file, indent=4)

    # Execute main processing function
    main(args.num_workers, all_prompts, args.output_file, args.api_key)