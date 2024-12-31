# Usage:
# python scripts/download_hf_dataset.py blanchon/EuroSAT_RGB

import argparse
import os

from datasets import load_dataset


def save_image(example, idx, output_dir, image_column):
    """Helper function to save a single image"""
    try:
        filename = f"image_{idx:06d}.png"
        filepath = os.path.join(output_dir, filename)
        example[image_column].save(filepath)
        return {"success": True}
    except Exception as e:
        print(f"Error processing image {idx}: {str(e)}")
        return {"success": False}


def download_dataset(dataset_id, output_dir, image_column="image", split="train"):
    """
    Download images from a Hugging Face dataset and save them locally.

    Args:
        dataset_id (str): The Hugging Face dataset ID (e.g., 'blanchon/EuroSAT_RGB')
        output_dir (str): Local directory to save the images
        image_column (str): The column name in the dataset that contains the images
        split (str): The dataset split to use (e.g., 'train', 'test', 'validation')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset: {dataset_id}")
    dataset = load_dataset(dataset_id)

    if split in dataset:
        data = dataset[split]
    else:
        available_splits = list(dataset.keys())
        print(
            f"Warning: Split '{split}' not found. Available splits: {available_splits}"
        )
        split = available_splits[0]
        print(f"Using '{split}' split instead.")
        data = dataset[split]

    print(f"Downloading {len(data)} images...")

    results = data.map(
        function=lambda example, idx: save_image(
            example, idx, output_dir, image_column
        ),
        with_indices=True,
        desc="Saving images",
        num_proc=8,
    )

    print(f"Download complete! Images saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download images from a Hugging Face dataset"
    )
    parser.add_argument(
        "dataset_id",
        type=str,
        help="Hugging Face dataset ID (e.g., 'blanchon/EuroSAT_RGB')",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="Column name in the dataset that contains the images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save the downloaded images",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (e.g., 'train', 'test', 'validation')",
    )

    args = parser.parse_args()
    download_dataset(args.dataset_id, args.output_dir, args.image_column, args.split)
