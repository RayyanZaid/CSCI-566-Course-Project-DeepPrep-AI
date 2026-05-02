import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.responseGen import generate_response
from backend.posture import analyze_posture_video
import cv2
import numpy as np

from backend.Tone.evaluateAudio import evaluateAudio, getInterviewResponseContentFeedback
from backend.FacialExpressionsAndEyeContact.inference import run_inference
def processVideo(video_path):
    """
    Process the video file to extract frames and perform analysis.
    """

    print(f"Processing video: {video_path}")

    # These should be replaced by the actual analysis functions that return scores from the Neural Nets. 
    # Input: "video_path"
    # Output: Scores and feedback 

    audio_interview_score, audio_overall_personality, audio_answer_score, audio_speaking_skills, audio_agreeableness, audio_conscientiousness, audio_neuroticism, audio_openness, audio_confidence_score = evaluateAudio(video_path)

    facialExpressionAndEyeContactModelPath = "backend/FacialExpressionsAndEyeContact/eye_contact_expression_v1_best.pt"
    eyeContactScore, facialAndEyeContactFeedback = run_inference(video_path, facialExpressionAndEyeContactModelPath)
    postureResult = analyze_posture_video(video_path)
    postureScore, postureFeedback = postureResult["score"], postureResult["feedback"]


    # Combine feedback from all analyses
    
    # Concatenate feedback into a single string

    # This is the prompt for LLM

    # Final Score

    # Scores are normalized from -1 to 1, we can to convert them to a 0 to 1 scale

    def scale_neg1_to_1_to_0_to_10(x: float) -> float:
        x = max(-1, min(1, x))
        
        return 5 * (x + 1)

    
    
    finalScore = (1/3) * scale_neg1_to_1_to_0_to_10(audio_interview_score) + (1/3) * scale_neg1_to_1_to_0_to_10(eyeContactScore) + (1/3) * scale_neg1_to_1_to_0_to_10(postureScore)

    finalModelFeedback = (
       f"Your final score is: {finalScore*100}%\n\n\n"
        f"Interview Response Content Feedback: {getInterviewResponseContentFeedback(audio_interview_score, audio_overall_personality, audio_answer_score, audio_speaking_skills, audio_agreeableness, audio_conscientiousness, audio_neuroticism, audio_openness, audio_confidence_score)}\n\n\n\n"
        f"Engagement (Eye Contact and Facial Expressions) Feedback: {facialAndEyeContactFeedback}\n\n\n\n"
        f"Body Language and Posture Feedback: EXAMPLE: If {postureScore} is lower than 0.6 than the interviee doesn't have good posture. Keep your analysis to 2 sentences. Here is some context feedback that could help with your analysis {postureFeedback}. All the scores here are normalized between 0 and 1 for different metrics of posture.")
        



    # Generate summary of the feedback
    fallback_summary = {
        "Final Score": f"{finalScore * 100:.1f}%",
        "Body Language and Posture Feedback": postureFeedback,
        "Engagement (Eye Contact and Facial Expressions) Feedback": eyeContactFeedback,
        "Interview Response Content Feedback": prosodyFeedback,
    }

    try:
        summary = generate_response(finalModelFeedback)
    except Exception as exc:
        print(f"LLM summary unavailable, using local fallback. Reason: {exc}")
        summary = json.dumps(fallback_summary)

    print("Summary of the interview:")
    print(summary)

    return summary

   
