import shutil

# Ignore this, just for me to get an example video for testing


for i in range(1, 10):
    src = f'/Users/rayyanzaid/.cache/huggingface/hub/datasets--AI4A-lab--RecruitView/snapshots/0cfa07ed0a43622f9104592b100d7bf3a25f6140/videos/vid_000{i}.mp4'
    dst = f'/Users/rayyanzaid/Desktop/School/CSCI-566-DeepLearning/CSCI-566-Course-Project-DeepPrep-AI/testVideos/vid_000{i}.mp4'
    
    shutil.copy(src, dst)

    print("Saved to:", dst)