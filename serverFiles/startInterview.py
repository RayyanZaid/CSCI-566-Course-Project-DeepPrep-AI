import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.responseGen import generate_response
import cv2
import numpy as np

from backend.Tone.evaluateAudio import evaluateAudio, getInterviewResponseContentFeedback
from backend.Engagement.evaluateEngagement import evaluateEngagement, getEngagementFeedback

def processVideo(video_path):
    """
    Process the video file to extract frames and perform analysis.
    """

    print(f"Processing video: {video_path}")

    # These should be replaced by the actual analysis functions that return scores from the Neural Nets. 
    # Input: "video_path"
    # Output: Scores and feedback 
    
    audio_interview_score, audio_overall_personality, audio_answer_score, audio_speaking_skills, audio_agreeableness, audio_conscientiousness, audio_neuroticism, audio_openness, audio_confidence_score = evaluateAudio(video_path)
    engagement_scores = evaluateEngagement(video_path)
    postureScore, postureFeedback = 0.5, "Posture was not good, you should sit up straighter and avoid slouching. Try to keep your shoulders back and maintain an open posture to appear more confident and engaged during the interview."

    # Combine feedback from all analyses
    
    # Concatenate feedback into a single string

    # This is the prompt for LLM

    # Final Score
    engagement_overall_score = engagement_scores.get("overall_performance", 0.0)
    finalScore = (1/3) * audio_interview_score + (1/3) * engagement_overall_score + (1/3) * postureScore

    finalModelFeedback = (
       f"Your final score is: {finalScore*100}%\n\n\n"
        f"Interview Response Content Feedback: {getInterviewResponseContentFeedback(audio_interview_score, audio_overall_personality, audio_answer_score, audio_speaking_skills, audio_agreeableness, audio_conscientiousness, audio_neuroticism, audio_openness, audio_confidence_score)}\n\n\n\n"
        f"Engagement (Eye Contact and Facial Expressions) Feedback: {getEngagementFeedback(engagement_scores)}\n\n\n\n"
        f"Body Language and Posture Feedback: EXAMPLE: If {postureScore} is lower than 0.6 than the interviee doesn't have good posture. Keep your analysis to 2 sentences. Here is some context feedback that could help with your analysis {postureFeedback}. All the scores here are normalized between 0 and 1 for different metrics of posture.")
        



    # Generate summary of the feedback

    # finalModelFeedback = "DO NOT WASTE ANY TOKENS, just return 'HI' for each of the 3 JSON values"
    summary = generate_response(finalModelFeedback)

    print("Summary of the interview:")
    print(summary)

    return summary

   