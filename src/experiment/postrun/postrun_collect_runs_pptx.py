import os
import shutil
import re
import json

def main():
    source_root = "runs_0724_gpt_final"
    target_folder = "results/pptx_0724_gpt_final"
    config_file_path = "doc2slide_dataset/config_0724_target150_template10/slidegen_config.json"
    output_config_file_path = f"{target_folder}/slidegen_config_filtered.json"
    
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    found_target_ids = set()

    for dir_name in os.listdir(source_root):
        source_dir_path = os.path.join(source_root, dir_name)
        if os.path.isdir(source_dir_path):
            match = re.search(r"_tgt(\d+)$", dir_name)
            if match:
                target_id = match.group(1)
                source_file = os.path.join(source_dir_path, "final.pptx")
                if os.path.exists(source_file):
                    target_file = os.path.join(target_folder, f"{target_id}.pptx")
                    shutil.copy(source_file, target_file)
                    print(f"Copied {source_file} to {target_file}")
                    found_target_ids.add(target_id)

    with open(config_file_path, "r") as f:
        config_data = json.load(f)

    original_count = len(config_data)

    filtered_config = [
        item for item in config_data
        if item.get("target", "").removesuffix(".pdf") in found_target_ids
    ]

    with open(output_config_file_path, "w") as f:
        json.dump(filtered_config, f, indent=4)

    print(f"\nFiltered {config_file_path} to {output_config_file_path}.")
    print(f"Kept {len(filtered_config)} items out of {original_count}.")


if __name__ == "__main__":
    main() 