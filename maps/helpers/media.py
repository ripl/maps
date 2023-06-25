#!/usr/bin/env python3
import numpy as np
import wandb

def save_video(frames: np.ndarray):
    # (height, width, channel) --> (channel, height, width)
    frames = np.asarray([np.asarray(f).transpose(2, 0, 1) for f in frames])
    wandb.log({'eval/video': wandb.Video(frames, fps=20, format="mp4")})
