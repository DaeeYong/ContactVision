import os
import numpy as np
import cv2

# joint → OpenPose keypoint index
JOINT_INDICES = {
    'left_toe': 19,
    'right_toe': 22,
    'left_heel': 21,
    'right_heel': 24,
}


#LABEL_ORDER = ['left_heel', 'left_toe', 'right_heel', 'right_toe']
LABEL_ORDER = ['left_toe', 'right_toe', 'left_heel', 'right_heel']

DEFAULT_COLOR = (0, 0, 255)
HIGHLIGHT_COLOR = (0, 255, 0)
MARKER_RADIUS = 10
MARKER_THICKNESS = -1

def process_video(video_path: str, pose_npy: str, label_npy: str, output_video: str) -> None:
    poses = np.load(pose_npy)   # shape (n_frames, 25, 3)
    poses = np.squeeze(poses)
    labels = np.load(label_npy) # shape (n_frames, 4)  ← [Rt, Lh, Lt, Rh]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = output_video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= poses.shape[0]:
            break

        joints       = poses[frame_idx]   # (25, 3)
        frame_labels = labels[frame_idx]  # (4,)

        for joint_name, op_idx in JOINT_INDICES.items():
            x, y, c = joints[op_idx]
            if c <= 0:
                continue
            
            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            label_idx = LABEL_ORDER.index(joint_name)

            if frame_labels[label_idx] == 1:
                color = HIGHLIGHT_COLOR
            else:
                color = DEFAULT_COLOR

            cv2.circle(frame, (int(x), int(y)), MARKER_RADIUS, color, MARKER_THICKNESS)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Processed video saved to: {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Overlay OpenPose markers on video and highlight based on labels.'
    )
    parser.add_argument('--video',  required=True, help='Path to input .mp4 video file')
    parser.add_argument('--pose',   required=True, help='Path to OpenPose .npy file (shape: n_frames,25,3)')
    parser.add_argument('--labels', required=True, help='Path to labels .npy file (shape: n_frames,4)')
    parser.add_argument('--out_path', required=True, help='Out Video Path')

    args = parser.parse_args()
    process_video(args.video, args.pose, args.labels, args.out_path)
    

