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

# (T, 25, 3) -> (T, 13, 3)
def get_lower_joints(target_npy):
    lower_idx = [8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24]
    return target_npy[ : , lower_idx, : ]

def preprocess_lower_joints(lower_joints, conf_threshold=0.3, max_interp_gap=3):
    F, K, _ = lower_joints.shape
    mask = lower_joints[:, :, 2] > conf_threshold  # (F, 13)

    pel_mask = mask[:, 0].copy()
    pel_coord = lower_joints[:, 0, :2].copy()      # (F, 2)

    valid_idxs = np.where(pel_mask)[0]
    if valid_idxs.size > 0:
        first = valid_idxs[0]
        pel_coord[:first] = pel_coord[first]
        pel_mask[:first] = True

        for i in range(len(valid_idxs)-1):
            s, e = valid_idxs[i], valid_idxs[i+1]
            gap = e - s - 1
            if 0 < gap <= max_interp_gap:
                for t in range(1, gap+1):
                    alpha = t / (gap+1)
                    pel_coord[s+t] = (1-alpha)*pel_coord[s] + alpha*pel_coord[e]
                    pel_mask[s+t] = True

        last = valid_idxs[-1]
        pel_coord[last+1:] = pel_coord[last]
        pel_mask[last+1:] = True

    # Transformation (relative coordinate, Pelvis)
    rel_data = lower_joints.astype(float).copy()
    rel_data[:, :, :2] -= pel_coord[:, None, :]

    for j in range(K):
        joint_mask = mask[:, j].copy()
        coord = rel_data[:, j, :2].copy()
        valid = np.where(joint_mask)[0]
        if valid.size == 0:
            continue

        first = valid[0]
        coord[:first] = coord[first]
        joint_mask[:first] = True

        for i in range(len(valid)-1):
            s, e = valid[i], valid[i+1]
            gap = e - s - 1
            if 0 < gap <= max_interp_gap:
                for t in range(1, gap+1):
                    alpha = t / (gap+1)
                    coord[s+t] = (1-alpha)*coord[s] + alpha*coord[e]
                    joint_mask[s+t] = True

        last = valid[-1]
        coord[last+1:] = coord[last]
        joint_mask[last+1:] = True

        rel_data[:, j, :2] = coord
        mask[:, j] = joint_mask

    return rel_data, mask
        
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
    
    lower_joints = get_lower_joints(output_npy)
    print(Fore.GREEN + f'lower_joints_shape: {lower_joints.shape}')

    processed, _ = preprocess_lower_joints(lower_joints)
    np.save(output_path, processed)
    print(Fore.GREEN + f'lower_joints_processed Done: {lower_joints.shape}')

    
if __name__ == "__main__":
    main()