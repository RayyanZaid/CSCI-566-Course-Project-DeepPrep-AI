# This file shows how to open a video file from the RecruitView Dataset

from datasets import load_dataset

dataset = load_dataset("AI4A-lab/RecruitView")

videoFile = dataset["train"][0]["video"]

import cv2

for i in range(videoFile._num_frames):
    frame = videoFile[i].data.permute(1, 2, 0).numpy()
    
    # OpenCV uses BGR, so we swap RGB to BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    cv2.imshow('Video Preview', frame_bgr)
    
    # Press 'q' to quit the video early
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()