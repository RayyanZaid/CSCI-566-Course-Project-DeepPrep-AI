#https://huggingface.co/google/gemma-7b
#pip install -U transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")

input_text = f"""
    Your task is to create a summary based on the information given in the following sentences about the interview. 
        This information will be about body language, seating posture, eye contact maintained, facial expressions, and the interview response content itself. 
        The summary should be in a professional tone and no more than 250 words. 
        Based on the content per behavior, determine whether it was good or ways they should improve that aspect. Here is the information: #insert_video_information_here. 
            "Return the feedback above in a json format like this:
            "Final Score": "",
            "Body Language and Posture Feedback": "",
            "Engagement (Eye Contact and Facial Expressions) Feedback": "",
            "Interview Response Content Feedback": ""

 """ 
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids, max_new_tokens=250)
print(tokenizer.decode(outputs[0]))


#dw abt it:
# deactivate
# py -3.11 -m venv csci-566-project-venv-311
# .\csci-566-project-venv-311\Scripts\Activate.ps1
# python --version
# python -m pip install --upgrade pip
# pip install -r requirements.txt
# pip install mediapipe==0.10.14
# python -c "import mediapipe as mp; print(mp.__version__); print(hasattr(mp, 'solutions'))"