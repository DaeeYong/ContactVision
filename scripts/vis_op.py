import cv2
import numpy as np
import argparse
from colorama import Fore, Style, init
init(autoreset=True)

def play_video(video_path, npy_path, output_path, save_flag=False):
    # 1) data load
    keypoints = np.load(npy_path)  
    # shape: (n_frames, n_keypoints, 3)

    # 2) open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    # origin video info
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # save option
    out = None
    if save_flag:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= keypoints.shape[0]:
            break
        
        person = keypoints[frame_idx]  # shape: (25, 3)
        for x, y, c in person:
            if np.isnan(x) or c < 0.1:
                continue
            cv2.circle(frame, (int(x), int(y)), 3, (0,255,0), 3)

        if save_flag and out is not None:
            out.write(frame)

        cv2.imshow('overlay', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    if save_flag and out is not None:
        out.release()

    cv2.destroyAllWindows()
    if save_flag:
        print(Fore.GREEN + f'Video Save Done: {output_path}')

def main():
    parser = argparse.ArgumentParser(description="OpenPose jsons to .npy")
    parser.add_argument('--input_npy', type=str, required=True, help='OpenPose npy(T, 25, 3)')
    parser.add_argument('--input_video', type=str, required=True, help='Target Video')
    parser.add_argument('--output_path', type=str, required=False, help='Output Video Path')
    parser.add_argument('--flag', type=int, default=1, help="Not Save: 0, Save: 1")
    
    args = parser.parse_args()
    
    input_npy = args.input_npy
    input_video = args.input_video
    output_path = args.output_path
    flag = args.flag
    
    play_video(input_video, input_npy, output_path, save_flag=flag)
    if flag == 1:
        print(Fore.GREEN + f'Video Save Done: {output_path}')

if __name__ == "__main__":
    main()
