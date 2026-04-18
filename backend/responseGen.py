#https://huggingface.co/google/gemma-7b
#pip install -U transformers


import os
from dotenv import load_dotenv

import google.generativeai as genai


load_dotenv()

# Note: This API Key is coming from the .env file. If you want to test the LLM:
# 1. Create a .env file in the root directory of the project
# 2. Add the line: API_KEY=get_from_rayyan

api_key = os.getenv('API_KEY')

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemma-3-27b-it")


def generate_response(video_information : str):
    

    input_text = f"""
        Your task is to create a summary based on the information given in the following sentences about the interview. 
            This information will be about body language, seating posture, eye contact maintained, facial expressions, and the interview response content itself. 
            The summary should be in a professional tone and no more than 250 words. 
            Based on the content per behavior, determine whether it was good or ways they should improve that aspect. Here is the information: {video_information} 
                "Return the feedback above in a json format like this:
                "Final Score": "",
                "Body Language and Posture Feedback": "",
                "Engagement (Eye Contact and Facial Expressions) Feedback": "",
                "Interview Response Content Feedback": ""

    """ 
    response = model.generate_content(input_text)
    return response.text


#dw abt it:
# deactivate
# py -3.11 -m venv csci-566-project-venv-311
# .\csci-566-project-venv-311\Scripts\Activate.ps1
# python --version
# python -m pip install --upgrade pip
# pip install -r requirements.txt
# pip install mediapipe==0.10.14
# python -c "import mediapipe as mp; print(mp.__version__); print(hasattr(mp, 'solutions'))"