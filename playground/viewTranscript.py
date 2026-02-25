from datasets import load_dataset

dataset = load_dataset("AI4A-lab/RecruitView")

transcriptOfFirstVideo = dataset["train"][0]["transcript"]


print(transcriptOfFirstVideo)