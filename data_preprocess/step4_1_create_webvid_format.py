import json
import argparse
import pandas as pd


def merge_json_files_with_transmit_status(caption_file, output_file):
    # Load caption data from JSON file
    with open(caption_file, 'r', encoding='utf-8') as file:
        caption_data = json.load(file)
    
    # Extracting data and adding is_transmit status
    data = [{
        'videoid': key, 
        'name': value['Video_GPT4_Caption'],
        'is_transmit': '1'  # N/A for videos not found in either category
    } for key, value in caption_data.items()]
    
    # Creating a DataFrame from the extracted data
    df = pd.DataFrame(data)
    
    # Saving the DataFrame as a CSV file
    df.to_csv(output_file, index=False)
    
    # Output the path to the saved CSV file
    return f"CSV file saved at: {output_file}"

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert GPT4V video captions JSON to CSV.")
    parser.add_argument("--caption_file_path", type=str, default="./3_2_final_useful_gpt_video_caption.json", help="Path to the input JSON caption file.")
    parser.add_argument("--output_csv_file_path", type=str, default="./all_clean_data.csv", help="Path to save the output CSV file.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Process the JSON and convert it to CSV
    merge_json_files_with_transmit_status(args.caption_file_path, args.output_csv_file_path)