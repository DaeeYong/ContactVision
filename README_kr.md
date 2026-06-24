# ContactVision: Learning Foot Contact from Video for Physically Plausible Gait Animation

[Daeyong Kim](https://github.com/DaeeYong), [Gyuseok Yi](https://github.com/yigyu), [Ri Yu](https://yul85.github.io/)

Ajou University, South Korea

![Teaser Image](figures/teaser.jpg)

ContactVision은 비디오에서 발과 지면의 접촉 상태를 추정하는 모델입니다. OpenPose로 추출한 2D body keypoint를 입력으로 사용해, 각 프레임마다 toe와 heel이 지면에 닿아 있는지 예측합니다.

모델 출력 label 순서는 다음과 같습니다.

```python
['left_toe', 'right_toe', 'left_heel', 'right_heel']
```

## 개요

Foot-ground contact 정보는 gait analysis, motion reconstruction, physically plausible character animation 등에 유용합니다. 하지만 force plate나 pressure mat 같은 장비 없이 정확한 contact label을 얻기는 어렵습니다.

ContactVision은 비디오 기반 foot contact 추정을 위해 다음 파이프라인을 제공합니다.

1. 입력 비디오에 대해 OpenPose BODY_25 모델을 실행합니다.
2. OpenPose JSON 파일을 NumPy 배열로 변환합니다.
3. Lower-body joint를 모델 입력 형식으로 전처리합니다.
4. Pretrained ContactVision 모델로 foot contact label을 추론합니다.
5. 필요한 경우 원본 비디오 위에 contact 결과를 시각화합니다.

## Repository 구조

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

이 프로젝트를 사용하려면 OpenPose BODY_25 pose estimation 결과가 필요합니다. OpenPose 설치 및 실행 방법은 공식 repository를 참고하세요.

[https://github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

스크립트에서 사용하는 Python dependency는 다음과 같습니다.

- `torch`
- `numpy`
- `opencv-python`
- `colorama`

이 프로젝트는 아래 conda 환경 조합에서 실행을 확인했습니다.

| Package | Tested version |
| --- | --- |
| Python | 3.10.13 |
| PyTorch | 2.5.1 |
| NumPy | 1.26.4 |
| OpenCV | 4.11.0 |
| Colorama | 0.4.6 |

다른 최신 버전에서도 동작할 수 있지만, Python 3.10과 위 버전 조합을 가장 안전한 기준 환경으로 권장합니다.

## 설치

Repository를 clone합니다.

```bash
git clone https://github.com/DaeeYong/ContactVision.git
cd ContactVision
```

### Option A: 기존 Conda 환경 사용

이미 프로젝트 실행에 사용하는 conda 환경이 있다면, 스크립트 실행 전에 해당 환경을 활성화합니다. 예를 들어 현재 사용하는 환경이 `motte`라면:

```bash
conda activate motte
```

필요한 패키지가 설치되어 있는지 확인합니다.

```bash
python -c "import torch, numpy, cv2, colorama; print('dependencies ok')"
```

### Option B: 새 Conda 환경 생성

새 환경을 만들고 싶다면 다음처럼 생성할 수 있습니다.

```bash
conda create -n contactvision python=3.10 -y
conda activate contactvision
```

PyTorch를 설치합니다. CPU 또는 Apple Silicon 환경에서는 기본 PyTorch 설치로 충분한 경우가 많습니다.

```bash
pip install torch
```

CUDA 환경에서는 CUDA 버전에 맞는 설치 명령을 PyTorch 공식 설치 가이드에서 확인해 사용하는 것을 권장합니다.

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

나머지 dependency를 설치합니다.

```bash
pip install numpy opencv-python colorama
```

`scripts/vis_op.py`가 `cv2.imshow()`를 사용하므로 `opencv-python`을 권장합니다. GUI가 없는 headless server에서는 `scripts/vis_labels.py`처럼 비디오 파일만 저장하는 스크립트는 동작할 수 있지만, OpenCV 창을 띄우는 기능은 사용할 수 없습니다.

Pretrained checkpoint는 기본적으로 다음 경로에 있어야 합니다.

```text
checkpoints/best_model.pth
```

아래 명령어들은 모두 프로젝트 루트 디렉토리에서 실행하는 것을 기준으로 합니다.

## Sample Data로 빠르게 실행하기

OpenPose JSON 파일을 raw pose array로 변환합니다.

```bash
python -m scripts.opjson2npy \
  --input_dir data/sample/openpose/4 \
  --output_path output/4_raw.npy
```

출력 파일의 shape은 `(T, 25, 3)`입니다. `T`는 frame 수이고, 각 keypoint는 `(x, y, confidence)` 값을 가집니다.

모델 입력을 위한 lower-body pose를 생성합니다.

```bash
python -m scripts.preprocess \
  --input_dir data/sample/openpose/4 \
  --output_path output/4_final.npy
```

출력 파일의 shape은 `(T, 13, 3)`입니다.

Foot contact label을 추론합니다.

```bash
python -m scripts.inference \
  --input_path output/4_final.npy \
  --output_path output/4_labels.npy
```

출력 label 파일의 shape은 `(T, 4)`입니다.

추론 결과를 sample video 위에 시각화합니다.

```bash
python -m scripts.vis_labels \
  --video data/sample/video/4.mp4 \
  --pose output/4_raw.npy \
  --labels output/4_labels.npy \
  --out_path output/4_contact.mp4
```

## Custom Video 사용법

### 1. OpenPose 실행

사용하려는 비디오에 대해 OpenPose를 실행하고 BODY_25 JSON 파일을 저장합니다. 결과 디렉토리에는 각 frame에 대응하는 `*_keypoints.json` 파일이 있어야 합니다.

예시 디렉토리 구조:

```text
data/my_video/openpose/
|-- my_video_000000000000_keypoints.json
|-- my_video_000000000001_keypoints.json
|-- my_video_000000000002_keypoints.json
`-- ...
```

현재 스크립트들은 OpenPose BODY_25 keypoint index를 기준으로 작성되어 있습니다.

### 2. JSON을 Raw Pose NPY로 변환

```bash
python -m scripts.opjson2npy \
  --input_dir data/my_video/openpose \
  --output_path output/my_video_raw.npy
```

출력:

```text
output/my_video_raw.npy  # shape: (T, 25, 3)
```

### 3. 모델 입력 전처리

```bash
python -m scripts.preprocess \
  --input_dir data/my_video/openpose \
  --output_path output/my_video_final.npy
```

출력:

```text
output/my_video_final.npy  # shape: (T, 13, 3)
```

전처리 과정에서는 lower-body joint를 선택하고 pelvis 기준 상대좌표로 변환합니다.

### 4. Inference 실행

```bash
python -m scripts.inference \
  --input_path output/my_video_final.npy \
  --output_path output/my_video_labels.npy
```

출력:

```text
output/my_video_labels.npy  # shape: (T, 4)
```

Label 순서는 다음과 같습니다.

```python
['left_toe', 'right_toe', 'left_heel', 'right_heel']
```

각 값은 binary contact state입니다.

- `1`: contact
- `0`: no contact

### 5. Contact Label 시각화

```bash
python -m scripts.vis_labels \
  --video data/my_video/my_video.mp4 \
  --pose output/my_video_raw.npy \
  --labels output/my_video_labels.npy \
  --out_path output/my_video_contact.mp4
```

시각화 결과에서는 네 개의 foot keypoint가 표시되며, contact로 예측된 frame이 강조됩니다.

## Utility Scripts

### `scripts/opjson2npy.py`

OpenPose JSON 파일들을 raw NumPy pose array로 변환합니다.

```bash
python -m scripts.opjson2npy \
  --input_dir <openpose_json_dir> \
  --output_path <raw_pose.npy>
```

입력:

```text
OpenPose BODY_25 JSON directory
```

출력:

```text
(T, 25, 3) NumPy array
```

### `scripts/preprocess.py`

OpenPose BODY_25 joint에서 lower-body joint를 선택하고, pelvis 기준 상대좌표로 변환해 모델 입력을 생성합니다.

```bash
python -m scripts.preprocess \
  --input_dir <openpose_json_dir> \
  --output_path <processed_pose.npy>
```

출력:

```text
(T, 13, 3) NumPy array
```

### `scripts/inference.py`

Pretrained ContactVision 모델로 foot contact label을 추론합니다.

```bash
python -m scripts.inference \
  --input_path <processed_pose.npy> \
  --output_path <labels.npy>
```

출력:

```text
(T, 4) NumPy array
```

### `scripts/vis_labels.py`

추론된 contact label을 원본 비디오 위에 overlay합니다.

```bash
python -m scripts.vis_labels \
  --video <input_video.mp4> \
  --pose <raw_pose.npy> \
  --labels <labels.npy> \
  --out_path <output_video.mp4>
```

### `scripts/vis_op.py`

OpenPose keypoint를 비디오 위에 시각화합니다.

```bash
python -m scripts.vis_op \
  --input_video <input_video.mp4> \
  --input_npy <raw_pose.npy> \
  --output_path <output_video.mp4> \
  --flag 1
```

`--flag 1`은 결과 비디오를 저장하고, `--flag 0`은 화면에만 표시합니다.

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

스크립트는 프로젝트 루트에서 module 형태로 실행하는 것을 권장합니다.

```bash
python -m scripts.inference \
  --input_path output/4_final.npy \
  --output_path output/4_labels.npy
```

아래처럼 파일 경로를 직접 지정해 실행하는 방식은 피하는 것이 좋습니다.

```bash
python scripts/inference.py
```

파일 경로로 직접 실행하면 Python import path에 프로젝트 루트가 포함되지 않아 `src` module을 찾지 못할 수 있습니다.

### OpenPose 결과가 없거나 잘못된 경우

다음 항목을 확인하세요.

- OpenPose가 BODY_25 모델로 실행되었는지 확인합니다.
- JSON 디렉토리에 각 frame의 `*_keypoints.json` 파일이 있는지 확인합니다.
- 비디오 frame 수와 pose frame 수가 맞는지 확인합니다.
- 각 OpenPose JSON에서 사용할 사람이 첫 번째 detected person인지 확인합니다.

## Citation

이 코드 또는 모델을 사용한다면 ContactVision paper를 인용해 주세요.

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

[LICENSE](LICENSE)를 참고하세요.
