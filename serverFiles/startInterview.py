import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.responseGen import generate_response
import cv2
import numpy as np

from backend.Tone.evaluateAudio import evaluateAudio, getInterviewResponseContentFeedback

def processVideo(video_path):
    """
    Process the video file to extract frames and perform analysis.
    """

    print(f"Processing video: {video_path}")

    # These should be replaced by the actual analysis functions that return scores from the Neural Nets. 
    # Input: "video_path"
    # Output: Scores and feedback 

    audio_interview_score, audio_overall_personality, audio_answer_score, audio_speaking_skills, audio_agreeableness, audio_conscientiousness, audio_neuroticism, audio_openness, audio_confidence_score = evaluateAudio(video_path)
    facialGestureScore, eyeContactFeedback, CURRENT_GOOD_EYE_CNT_PCT = 1, "Eye contact was maintained", 0.7
    postureScore, postureFeedback = 0.5, "Posture was not good, you should sit up straighter and avoid slouching. Try to keep your shoulders back and maintain an open posture to appear more confident and engaged during the interview."

    # Combine feedback from all analyses
    
    # Concatenate feedback into a single string

    # This is the prompt for LLM

    # Final Score
    finalScore = (1/3) * audio_interview_score + (1/3) * facialGestureScore + (1/3) * postureScore

    finalModelFeedback = (
       f"Your final score is: {finalScore*100}%\n\n\n"
        f"Interview Response Content Feedback: {getInterviewResponseContentFeedback(audio_interview_score, audio_overall_personality, audio_answer_score, audio_speaking_skills, audio_agreeableness, audio_conscientiousness, audio_neuroticism, audio_openness, audio_confidence_score)}\n\n\n\n"
        f"Engagement (Eye Contact and Facial Expressions) Feedback: EXAMPLE: For facial gestures, the interviewee scored a {facialGestureScore/1} Give feedback based on score. For eyeContact, if {eyeContactFeedback} is within 15 of {CURRENT_GOOD_EYE_CNT_PCT}, then the interviee is maintaing a good level of eye contact. Depending on whether they are too far above or below the range, give tips accordingly. To improve eye contact say things like to be more relaxed and comfortable. That it's okay to glance away when speaking or thinking, but make sure to look at them when listening and making important points. Things in that nature. If the {eyeContactFeedback} is within 15 of {CURRENT_GOOD_EYE_CNT_PCT}, then state that their eye contact maintence was good and no improvements are needed there. Make your response brief (2 sentences max) and natural, like a reviewer or teacher. \n\n\n\n"
        f"Body Language and Posture Feedback: EXAMPLE: If {postureScore} is lower than 0.6 than the interviee doesn't have good posture. Keep your analysis to 2 sentences. Here is some context feedback that could help with your analysis {postureFeedback}. All the scores here are normalized between 0 and 1 for different metrics of posture.")
        



    # Generate summary of the feedback

    # finalModelFeedback = "DO NOT WASTE ANY TOKENS, just return 'HI' for each of the 3 JSON values"
    summary = generate_response(finalModelFeedback)

    print("Summary of the interview:")
    print(summary)

    return summary

   