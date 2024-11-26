#!/usr/bin/env python3
from PIL import Image
import numpy as np
import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def find_bounds(img_array):
    """Find boundaries of non-transparent content using NumPy operations."""
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # Use alpha channel for RGBA images
        mask = img_array[:, :, 3] > 0
    else:
        # For RGB images, detect non-white pixels
        mask = np.any(img_array[:, :, :3] < 255, axis=2)
    
    # Find coordinates of non-transparent pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    # Get min/max coordinates
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    
    return (left, top, right + 1, bottom + 1)

def trim_image(img_path, output_path):
    """Process a single image."""
    try:
        # Open image and convert to NumPy array
        with Image.open(img_path) as img:
            # Convert P mode images with transparency to RGBA
            if img.mode == 'P' and 'transparency' in img.info:
                img = img.convert('RGBA')
            
            # Convert image to numpy array
            img_array = np.array(img)
            
            # Find bounds
            bounds = find_bounds(img_array)
            
            if bounds is None:
                print(f"Warning: No content found in {img_path}")
                return
            
            # Crop image
            cropped = img.crop(bounds)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with original format and transparency
            cropped.save(output_path, quality=95)
            
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")

def process_directory(input_dir, output_dir, recursive=False, max_workers=None):
    """Process all images in directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Get list of image files
    extensions = ('.png', '.jpg', '.jpeg', '.gif', '.PNG', '.JPG', '.JPEG', '.GIF')
    if recursive:
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.rglob(f'*{ext}'))
    else:
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
    
    # Prepare processing tasks
    tasks = []
    for img_path in image_files:
        rel_path = img_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        tasks.append((img_path, output_path))
    
    # Process images in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(
            executor.map(lambda x: trim_image(*x), tasks),
            total=len(tasks),
            desc="Processing images"
        ))

def main():
    parser = argparse.ArgumentParser(description='Trim whitespace from images while preserving transparency')
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('output_dir', help='Output directory for processed images')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Process subdirectories recursively')
    parser.add_argument('-w', '--workers', type=int, default=None,
                        help='Number of worker threads (default: CPU count)')
    
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, args.recursive, args.workers)

if __name__ == '__main__':
    main()