#!/usr/bin/env python3
"""
Script to load and save the Unlearning/rmu-training-data dataset.
Fixes schema mismatch errors by loading specific data files.
"""

from datasets import load_dataset
from huggingface_hub import list_repo_files
import os
import fnmatch

def main():
    dataset_name = "Unlearning/rmu-training-data"
    
    print(f"Inspecting dataset repository: {dataset_name}")
    print("=" * 80)

    try:
        # List all files in the Hugging Face repository
        all_files = list_repo_files(repo_id=dataset_name, repo_type="dataset")
        data_files_list = [f for f in all_files if f.endswith('.json') or f.endswith('.jsonl') or f.endswith('.parquet')]
        
        print("Found data files:")
        for f in data_files_list:
            print(f"  - {f}")
        print("-" * 80)

        # Define the specific file mapping we want to load
        # Based on your request for 'bio_forget_corpus'
        target_file = "bio-forget-corpus.json"
        
        if target_file not in data_files_list:
            # Fallback: try to find a file that matches the name partially
            matches = fnmatch.filter(data_files_list, "*bio*forget*")
            if matches:
                target_file = matches[0]
            else:
                print(f"Error: Could not find a file matching '{target_file}'")
                return

        print(f"\nLoading specific data file: {target_file}")
        
        # Load ONLY the specific file to avoid schema mismatch
        # We assign it to the 'train' split since it's a raw file load
        dataset = load_dataset(dataset_name, data_files=target_file)
        
        print(f"\nDataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Inspect the loaded data
        split_name = list(dataset.keys())[0]
        split = dataset[split_name]
        print(f"\nDetails for file '{target_file}':")
        print(f"  Number of examples: {len(split)}")
        print(f"  Features: {list(split.features.keys())}")

        print("\n" + "=" * 80)
        print("Do you want to save this subset to disk? (yes/no)")
        response = input().strip().lower()

        if response in ['yes', 'y']:
            output_dir = "rmu_training_data"
            os.makedirs(output_dir, exist_ok=True)
            
            # Construct a safe save name from the filename (e.g., bio-forget-corpus)
            save_name = os.path.splitext(target_file)[0]
            output_path = os.path.join(output_dir, save_name)

            print(f"\nSaving to {output_path}...")
            dataset.save_to_disk(output_path)
            print(f"✓ Saved {len(split)} examples to {output_path}")

    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()