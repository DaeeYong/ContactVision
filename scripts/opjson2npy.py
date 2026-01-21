import argparse
import os
import glob
import numpy as np
import json

from colorama import Fore, Style, init
init(autoreset=True)

# OpenPose jsons to npy
def jsons_to_npy(json_dir: str, pad_value: float = np.nan):

    json_paths = sorted(glob.glob(os.path.join(json_dir, '*.json')))
    if not json_paths:
        raise RuntimeError(f"No JSON files found in {json_dir!r}")

    n_keypoints = 25

    n_frames = len(json_paths)
    result = np.full((n_frames, n_keypoints, 3),
                    pad_value, dtype=np.float32)

    for i, path in enumerate(json_paths):
        with open(path, 'r') as f:
            js = json.load(f)
        people = js.get('people', [])
        if not people:
            continue    
        coords = np.array(people[0].get('pose_keypoints_2d', []),
                            dtype=np.float32).reshape(-1, 3)
        if coords.shape[0] != n_keypoints:
            padded = np.full((n_keypoints, 3), pad_value, dtype=np.float32)
            valid = min(coords.shape[0], n_keypoints)
            padded[:valid] = coords[:valid]
            coords = padded
        
        result[i] = coords

    return result

def main():
    parser = argparse.ArgumentParser(description="OpenPose jsons to .npy")
    parser.add_argument('--input_dir', type=str, required=True, help='OpenPose jsons dir')
    parser.add_argument('--output_path', type=str, required=True, help='Output .npy path')
    
    args = parser.parse_args()
    json_dir = args.input_dir
    output_path = args.output_path
    
    # OpenPose jsons -> .npy (T, 25, 3)
    output_npy = jsons_to_npy(json_dir)
    print(Fore.GREEN + f'output_npy shape: {output_npy.shape}')

    np.save(output_path, output_npy)

if __name__ == "__main__":
    main()