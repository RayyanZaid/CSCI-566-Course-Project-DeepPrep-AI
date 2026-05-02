import os
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()

api_key = os.getenv('API_KEY')

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemma-3-27b-it")


def generate_response(video_information: str):
    input_text = f"""
    Your task is to create a summary based on the information given in the following sentences about the interview.
    This information will be about body language, seating posture, eye contact maintained, facial expressions, and the interview response content itself.
    The summary should be in a professional tone and no more than 250 words.
    Based on the content per behavior, determine whether it was good or ways they should improve that aspect. Here is the information: {video_information}
    Return the feedback above in a JSON object with the keys: "Final Score", "Body Language and Posture Feedback", "Engagement (Eye Contact and Facial Expressions) Feedback", and "Interview Response Content Feedback".
    """
    response = model.generate_content(input_text)
    return response.text