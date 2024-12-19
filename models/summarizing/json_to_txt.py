import os
import json

def convert_json_to_txt(json_dir, txt_dir):
    """
    Convert JSON files into text files by concatenating all sentences and removing topic labels.
    """
    # Ensure the output directory exists
    os.makedirs(txt_dir, exist_ok=True)

    # Iterate through all JSON files in the directory
    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json"):
            # Construct full file paths
            json_file_path = os.path.join(json_dir, file_name)
            txt_file_name = file_name.replace(".json", ".txt")
            txt_file_path = os.path.join(txt_dir, txt_file_name)

            # Read the JSON file
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)

            # Concatenate all sentences into a single string
            concatenated_text = " ".join(data.values())

            # Write to the .txt file
            with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(concatenated_text)

            print(f"Converted {file_name} to {txt_file_name}")

# Define directories
json_directory = "./summarized_lectures_by_topic/C"  # Replace with your JSON files directory
txt_directory = "./summarized_lectures"    # Replace with your desired output directory

# Run the conversion
convert_json_to_txt(json_directory, txt_directory)
