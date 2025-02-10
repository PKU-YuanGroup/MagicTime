import re
import json
import argparse


def process_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    matched_data = {}
    unmatched_data = {}

    for key, value in data.items():
        video_summary_match = re.search(r'"Video_Summary": (.*)', value)

        if video_summary_match:
            matched_data[key] = {
                "Video_GPT4_Caption": video_summary_match.group(1),
            }
        else:
            unmatched_data[key] = value

    return matched_data, unmatched_data

def read_json_file(file_path):
    """Reads a JSON file and returns its content."""
    with open(file_path, 'r') as file:
        return json.load(file)

def remove_by_Frame(data):
    # Initialize dictionaries for matched (to exclude) and unmatched data
    to_exclude = {}
    to_keep = {}

    # Pattern to identify "by Frame X" in the video summary
    pattern = re.compile(r'(by|at|in|on) Frame \d+', re.IGNORECASE)

    for key, value in data.items():
        # Assuming "Video_Summary" is a direct key in the value dictionary
        video_summary = value.get("Video_GPT4_Caption", "")
        # Check if "by Frame X" is in the video summary
        if pattern.search(video_summary):
            to_exclude[key] = value
        else:
            to_keep[key] = value

    return to_keep, to_exclude

def remove_unmatch_records(gpt_data, unmatched_json_data):
    """
    Removes records from gpt_results if their ID exists in disordered_records.
    :param gpt_data: dict, the data from gpt_results.json
    :param disordered_ids: set, the set of IDs from disordered_records.json
    :return: dict, the updated gpt_data with matching records removed
    """
    disordered_ids = set(unmatched_json_data.keys())
    return {id_: value for id_, value in gpt_data.items() if id_ not in disordered_ids}

def save_json_file(data, file_path):
    """Saves data to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def merge_json_files(info_data, caption_data):
    # Merge info into caption data based on matching key prefixes
    for caption_key in caption_data:
        for info_key in info_data:
            if caption_key.startswith(info_key):
                selected_info = {key: info_data[info_key][key] for key in ['title'] if
                                 key in info_data[info_key]}
                caption_data[caption_key].update(selected_info)
                
                break
    return caption_data

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process GPT4V video captions and clean up data.")
    parser.add_argument("--file_path", type=str, default="./3_1_gpt_video_caption.json", help="Path to the input JSON file.")
    parser.add_argument("--updated_file_path", type=str, default="./3_1_gpt_video_caption.json", help="Path to save the updated JSON file.")
    parser.add_argument("--unmatched_data_path", type=str, default="./3_2_temp_unmatched_gpt_video_caption.json", help="Path to save unmatched records.")
    parser.add_argument("--exclude_by_frame_data_path", type=str, default="./3_2_temp_exclude_by_frame_gpt_video_caption.json", help="Path to save excluded records.")
    parser.add_argument("--final_useful_data_path", type=str, default="./3_2_final_useful_gpt_video_caption.json", help="Path to save the final cleaned data.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Processing steps
    matched_data, unmatched_data = process_json(args.file_path)
    to_keep, to_exclude = remove_by_Frame(matched_data)

    # Clean JSON by removing unmatched and excluded records
    updated_json = remove_unmatch_records(remove_unmatch_records(read_json_file(args.file_path), unmatched_data), to_exclude)

    # Save intermediate results
    save_json_file(unmatched_data, args.unmatched_data_path)
    save_json_file(to_exclude, args.exclude_by_frame_data_path)
    save_json_file(to_keep, args.final_useful_data_path)

    # Print stats
    if len(unmatched_data) != 0 or len(to_exclude) != 0:
        save_json_file(updated_json, args.updated_file_path)
        print(f"Found {len(unmatched_data)} unmatched_data and {len(to_exclude)} exclude_by_frame_data!")
        print(f"Updated JSON file has been saved to {args.updated_file_path}. Please rerun GPT4V for captioning.")
    else:
        print(f"No unmatched_data and exclude_by_frame_data found! You can directly use {args.final_useful_data_path} for the next step.")