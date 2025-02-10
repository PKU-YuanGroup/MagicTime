import re
import json
import argparse

def load_json(file_path):
    """Load and return the content of a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def process_frame_caption(file_path):
    """Process frame captions and save matched data."""
    data = load_json(file_path)
    matched_data = {}
    unmatched_data = {}
    for key, value in data.items():
        brief_reasoning_match = re.search(r'Brief Reasoning Statement: (.*?)(?:\n\n|\n)', value, re.DOTALL)
        reasoning_match = re.search(r'"Reasoning": \[(.*?)\]', value, re.DOTALL)
        captioning_match = re.search(r'"Captioning": \[(.*?)\]', value, re.DOTALL)
        if brief_reasoning_match and reasoning_match and captioning_match:
            brief_reasoning = brief_reasoning_match.group(1).strip()
            reasoning_list = re.findall(r'"(.*?)"(?:,|$)', reasoning_match.group(1))
            captioning_list = re.findall(r'"(.*?)"(?:,|$)', captioning_match.group(1))
            matched_data[key] = {
                "Video_Reasoning": brief_reasoning,
                "Frame_Reasoning": reasoning_list,
                "Frame_Captioning": captioning_list
            }
        else:
            unmatched_data[key] = value
    return matched_data, unmatched_data

def is_disordered(section):
    frames = []
    for entry in section:
        try:
            # Extracting the frame number
            frame_num = int(entry.split(':')[0].split(' ')[1])
            frames.append(frame_num)
        except ValueError:
            # If parsing fails, skip this entry
            continue
    return not all(earlier <= later for earlier, later in zip(frames, frames[1:]))

def find_disorder(data):
    """Identify entries with unordered frames."""
    unordered_records = {}
    ordered_records = {}
    
    for key, value in data.items():
        for section_name in ['Frame_Reasoning', 'Frame_Captioning']:
            section = value.get(section_name, [])
            if is_disordered(section):
                unordered_records[key] = value
                break
            else:
                ordered_records[key] = value
    return ordered_records, unordered_records

def remove_disorder(data, unordered_data):
    """Remove disordered entries from the dataset."""
    unordered_ids = set(unordered_data.keys())
    ordered_json = {k: v for k, v in data.items() if k not in unordered_ids}
    return ordered_json

def remove_unmatch_records(data, unmatched_data):
    """
    Removes records from gpt_results if their ID exists in disordered_records.
    :param data: dict, the data from gpt_results.json
    :return: dict, the updated data with matching records removed
    """
    unmatch_ids = set(unmatched_data.keys())
    matched_json = {id_: value for id_, value in data.items() if id_ not in unmatch_ids}
    return matched_json

def merge_json_files(info_data, caption_data):
    # Load info and caption data from JSON files
    # with open(info_file, 'r') as file:
    #     info_data = json.load(file)
    # with open(caption_file, 'r') as file:
    #     caption_data = json.load(file)

    # Merge info into caption data based on matching key prefixes
    for caption_key in caption_data:
        for info_key in info_data:
            if caption_key.startswith(info_key):
                # Update the caption entry with info data
                
                # caption_data[caption_key].update(info_data[info_key])

                selected_info = {key: info_data[info_key][key] for key in ['title'] if
                                 key in info_data[info_key]}
                caption_data[caption_key].update(selected_info)
                
                break

    # Save merged data to a new JSON file
    # with open(output_file, 'w') as file:
    #     json.dump(caption_data, file)
    return caption_data

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process GPT4V frame captions and clean up data.")
    parser.add_argument("--file_path", type=str, default="./2_1_gpt_frames_caption.json", help="Path to the input JSON file.")
    parser.add_argument("--updated_file_path", type=str, default="./2_2_updated_gpt_frames_caption.json", help="Path to save the updated JSON file.")
    parser.add_argument("--unmatched_file_path", type=str, default="./2_2_temp_unmatched_gpt_frames_caption.json", help="Path to save unmatched records.")
    parser.add_argument("--unordered_file_path", type=str, default="./2_2_temp_unordered_gpt_frames_caption.json", help="Path to save unordered records.")
    parser.add_argument("--final_useful_data_file_path", type=str, default="./2_2_final_useful_gpt_frames_caption.json", help="Path to save the final cleaned data.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Processing steps
    matched_data, unmatched_data = process_frame_caption(args.file_path)
    ordered_records, unordered_records = find_disorder(matched_data)

    # Clean JSON by removing unmatched and unordered records
    updated_json = remove_unmatch_records(remove_disorder(load_json(args.file_path), unordered_records), unmatched_data)

    # Final useful data (can be merged with additional info if needed)
    final_useful_data = ordered_records

    # Print stats
    print(f"Number of Unmatched Records (GPT4V_Frame): {len(unmatched_data)}")
    print(f"Number of Unordered Records (GPT4V_Frame): {len(unordered_records)}")
    print(f"Number of Final Useful Records (GPT4V_Frame): {len(final_useful_data)}")

    # Save the processed results
    if len(unmatched_data) != 0 or len(unordered_records) != 0:
        save_json(updated_json, args.updated_file_path)
        print(f"Found {len(unmatched_data)} unmatched records and {len(unordered_records)} unordered records!")
        print(f"Updated JSON file has been saved to {args.updated_file_path}. Please rerun GPT4V for captioning.")
    else:
        print(f"No unmatched/unordered records found! You can directly use {args.final_useful_data_file_path} for the next step.")

    # Save intermediate results
    save_json(unmatched_data, args.unmatched_file_path)
    save_json(unordered_records, args.unordered_file_path)
    save_json(final_useful_data, args.final_useful_data_file_path)