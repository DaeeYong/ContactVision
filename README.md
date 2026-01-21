# ContactVision: Learning Foot Contact from Video for Physically Plausible Gait Animation
[Daeyong Kim](https://github.com/DaeeYong), [Gyuseok Yi](https://github.com/yigyu) and [Ri Yu](https://yul85.github.io/)
![Teaser Image](figures/teaser.jpg)

Ajou University, South Korea

## Abstract 
Foot-ground contact information plays a crucial role in character animation and gait analysis, as it helps accurately simulating
realistic movement patterns and understanding the biomechanics of walking. Existing motion datasets do not explicitly include
foot-ground contact information, requiring separate computation or manual annotation. Obtaining accurate foot-ground con-
tact information typically requires additional sensors such as pressure mats or force plates. Without such devices, estimating
contact becomes a highly challenging task. We propose ContactVision, a deep learning framework that detects heel and toe
contact states directly from video. Our network is trained in a supervised manner using contact labels derived from motion
capture data via ground reaction force estimation. This enables training on existing datasets without the need for additional
hardware. We demonstrate the utility of our contact detection network in two downstream tasks: gait motion reconstruction
and gait analysis. For animation, we incorporate predicted contact labels into a reinforcement learning framework with a two-
segment foot model, enabling realistic foot articulation behavior. For analysis, we estimate clinically relevant gait parameters
such as double and single support times, and validate the accuracy against pressure sensor mat data and prior video-based
methods. Our results show competitive performance in both animation and analysis settings. The code is publicly available at github.com/DaeeYong/ContactVision.

---
## Requirements
비디오의 입력으로 OpenPose의 결과물이 필요하다. OpenPose는 해당 깃허브 페이지를 참조.[https://github.com/CMU-Perceptual-Computing-Lab/openpose]


## Installation
## Steps for running with a custom video
## Run examples
모델의 출력은 ['left_toe', 'right_toe', 'left_heel', 'right_heel'] 이다.
## Bibtex