# ContactVision: Learning Foot Contact from Video for Physically Plausible Gait Animation

[Daeyong Kim](https://github.com/DaeeYong), [Gyuseok Yi](https://github.com/yigyu), [Ri Yu](https://yul85.github.io/)

Ajou University, South Korea

![Teaser Image](figures/teaser.jpg)

ContactVision predicts foot-ground contact states from video. It uses 2D body keypoints extracted with OpenPose as input and predicts whether each toe and heel is touching the ground for every frame.

The model output label order is:

```python
['left_toe', 'right_toe', 'left_heel', 'right_heel']
```

## Overview

Foot-ground contact information is useful for gait analysis, motion reconstruction, and physically plausible character animation. However, accurate contact labels are difficult to obtain without dedicated equipment such as force plates or pressure mats.

ContactVision provides the following pipeline for video-based foot contact estimation:

1. Run OpenPose BODY_25 on the input video.
2. Convert OpenPose JSON files into a NumPy array.
3. Preprocess lower-body joints into the model input format.
4. Run inference with the pretrained ContactVision model.
5. Optionally visualize the predicted contact states on the original video.

## Repository Structure

```text
ContactVision/
|-- checkpoints/
|   `-- best_model.pth
|-- data/
|   `-- sample/
|-- figures/
|-- scripts/
|   |-- opjson2npy.py
|   |-- preprocess.py
|   |-- inference.py
|   |-- vis_labels.py
|   `-- vis_op.py
`-- src/
    `-- model.py
```

## Requirements

This project requires OpenPose BODY_25 pose estimation results. For OpenPose installation and usage, see the official repository:

[https://github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

The Python dependencies used by the scripts are:

- `torch`
- `numpy`
- `opencv-python`
- `colorama`

This project has been tested with the following conda environment:

| Package | Tested version |
| --- | --- |
| Python | 3.10.13 |
| PyTorch | 2.5.1 |
| NumPy | 1.26.4 |
| OpenCV | 4.11.0 |
| Colorama | 0.4.6 |

Other recent versions may also work, but Python 3.10 with the versions above is the recommended baseline.

## Installation

Clone the repository:

```bash
git clone https://github.com/DaeeYong/ContactVision.git
cd ContactVision
```

### Option A: Use an Existing Conda Environment

If you already have a conda environment for this project, activate it before running the scripts. For example, if your environment is named `motte`:

```bash
conda activate motte
```

Check that the required packages are installed:

```bash
python -c "import torch, numpy, cv2, colorama; print('dependencies ok')"
```

### Option B: Create a New Conda Environment

To create a new environment:

```bash
conda create -n contactvision python=3.10 -y
conda activate contactvision
```

Install PyTorch. For CPU or Apple Silicon environments, the default PyTorch installation is often sufficient:

```bash
pip install torch
```

For CUDA environments, use the installation command that matches your CUDA version from the official PyTorch installation guide:

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Install the remaining dependencies:

```bash
pip install numpy opencv-python colorama
```

`scripts/vis_op.py` uses `cv2.imshow()`, so `opencv-python` is recommended. On a headless server without GUI support, scripts that only write video files, such as `scripts/vis_labels.py`, may still work, but OpenCV window display features will not be available.

The pretrained checkpoint is expected at:

```text
checkpoints/best_model.pth
```

All commands below assume they are run from the project root directory.

## Quick Start with Sample Data

Convert OpenPose JSON files into a raw pose array:

```bash
python -m scripts.opjson2npy \
  --input_dir data/sample/openpose/4 \
  --output_path output/4_raw.npy
```

The output shape is `(T, 25, 3)`, where `T` is the number of frames and each keypoint stores `(x, y, confidence)`.

Create the lower-body pose input for the model:

```bash
python -m scripts.preprocess \
  --input_dir data/sample/openpose/4 \
  --output_path output/4_final.npy
```

The output shape is `(T, 13, 3)`.

Run foot contact label inference:

```bash
python -m scripts.inference \
  --input_path output/4_final.npy \
  --output_path output/4_labels.npy
```

The output label file shape is `(T, 4)`.

Visualize the inference result on the sample video:

```bash
python -m scripts.vis_labels \
  --video data/sample/video/4.mp4 \
  --pose output/4_raw.npy \
  --labels output/4_labels.npy \
  --out_path output/4_contact.mp4
```

## Using a Custom Video

### 1. Run OpenPose

Run OpenPose on your target video and save BODY_25 JSON files. The output directory should contain one `*_keypoints.json` file for each frame.

Example directory structure:

```text
data/my_video/openpose/
|-- my_video_000000000000_keypoints.json
|-- my_video_000000000001_keypoints.json
|-- my_video_000000000002_keypoints.json
`-- ...
```

The current scripts are written for the OpenPose BODY_25 keypoint index format.

### 2. Convert JSON to Raw Pose NPY

```bash
python -m scripts.opjson2npy \
  --input_dir data/my_video/openpose \
  --output_path output/my_video_raw.npy
```

Output:

```text
output/my_video_raw.npy  # shape: (T, 25, 3)
```

### 3. Preprocess Model Input

```bash
python -m scripts.preprocess \
  --input_dir data/my_video/openpose \
  --output_path output/my_video_final.npy
```

Output:

```text
output/my_video_final.npy  # shape: (T, 13, 3)
```

The preprocessing step selects lower-body joints and converts them to pelvis-relative coordinates.

### 4. Run Inference

```bash
python -m scripts.inference \
  --input_path output/my_video_final.npy \
  --output_path output/my_video_labels.npy
```

Output:

```text
output/my_video_labels.npy  # shape: (T, 4)
```

The label order is:

```python
['left_toe', 'right_toe', 'left_heel', 'right_heel']
```

Each value is a binary contact state:

- `1`: contact
- `0`: no contact

### 5. Visualize Contact Labels

```bash
python -m scripts.vis_labels \
  --video data/my_video/my_video.mp4 \
  --pose output/my_video_raw.npy \
  --labels output/my_video_labels.npy \
  --out_path output/my_video_contact.mp4
```

The visualization marks the four foot keypoints and highlights frames where contact is predicted.

## Utility Scripts

### `scripts/opjson2npy.py`

Converts OpenPose JSON files into a raw NumPy pose array.

```bash
python -m scripts.opjson2npy \
  --input_dir <openpose_json_dir> \
  --output_path <raw_pose.npy>
```

Input:

```text
OpenPose BODY_25 JSON directory
```

Output:

```text
(T, 25, 3) NumPy array
```

### `scripts/preprocess.py`

Selects lower-body joints from OpenPose BODY_25 and converts them to pelvis-relative coordinates for model input.

```bash
python -m scripts.preprocess \
  --input_dir <openpose_json_dir> \
  --output_path <processed_pose.npy>
```

Output:

```text
(T, 13, 3) NumPy array
```

### `scripts/inference.py`

Predicts foot contact labels with the pretrained ContactVision model.

```bash
python -m scripts.inference \
  --input_path <processed_pose.npy> \
  --output_path <labels.npy>
```

Output:

```text
(T, 4) NumPy array
```

### `scripts/vis_labels.py`

Overlays predicted contact labels on the original video.

```bash
python -m scripts.vis_labels \
  --video <input_video.mp4> \
  --pose <raw_pose.npy> \
  --labels <labels.npy> \
  --out_path <output_video.mp4>
```

### `scripts/vis_op.py`

Visualizes OpenPose keypoints on a video.

```bash
python -m scripts.vis_op \
  --input_video <input_video.mp4> \
  --input_npy <raw_pose.npy> \
  --output_path <output_video.mp4> \
  --flag 1
```

`--flag 1` saves the output video, while `--flag 0` only displays the result on screen.

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run scripts as modules from the project root directory:

```bash
python -m scripts.inference \
  --input_path output/4_final.npy \
  --output_path output/4_labels.npy
```

Avoid running scripts directly by file path:

```bash
python scripts/inference.py
```

When a script is run directly by file path, the project root may not be included in Python's import path, which can prevent Python from finding the `src` module.

### Missing or Incorrect OpenPose Results

Check the following:

- OpenPose was run with the BODY_25 model.
- The JSON directory contains one `*_keypoints.json` file for each frame.
- The number of video frames matches the number of pose frames.
- The first detected person in each OpenPose JSON file is the intended subject.

## Citation

If you use this code or model, please cite the ContactVision paper:

```bibtex
@inproceedings{kim2026contactvision,
  title={ContactVision: Learning Foot Contact from Video for Physically Plausible Gait Animation},
  author={Kim, Daeyong and Yi, Gyuseok and Yu, Ri},
  booktitle={Computer Graphics Forum},
  pages={e70334},
  year={2026},
  organization={Wiley Online Library}
}
```

## License

See [LICENSE](LICENSE).
