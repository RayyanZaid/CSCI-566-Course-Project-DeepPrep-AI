import os
import subprocess
from backend.Engagement.inference import run_inference

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "Engagement", "eye_contact_expression_v1_best.pt")


def evaluateEngagement(video_path: str):
    """
    Runs engagement model and returns key scores.
    Mirrors evaluateAudio() style.
    """
    root_name, _ = os.path.splitext(video_path)
    converted_path = f"{root_name}_engagement_converted.mp4"

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-pix_fmt", "yuv420p",
        converted_path,
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    report = run_inference(
        video_path=converted_path,
        checkpoint_path=CHECKPOINT_PATH
    )

    metrics = report.get("metrics", {})

    scores = {}
    for key, val in metrics.items():
        if isinstance(val, dict) and "score" in val:
            scores[key] = float(val["score"])

    return scores


def getEngagementFeedback(scores: dict):
    """
    Builds prompt text for LLM (same pattern as Tone).
    """

    extraversion = scores.get("extraversion", 0.0)
    confidence_score = scores.get("confidence_score", 0.0)
    facial_expression = scores.get("facial_expression", 0.0)
    overall_performance = scores.get("overall_performance", 0.0)

    return (
        f"You are an expert interview coach providing engagement feedback based on video analysis.\n\n"
        f"All scores are standardized model outputs where values closer to 0 or above are better than strongly negative values.\n\n"
        f"Candidate engagement-related scores:\n"
        f"- Extraversion / expressiveness: {extraversion}\n"
        f"- Confidence: {confidence_score}\n"
        f"- Facial expression: {facial_expression}\n"
        f"- Overall engagement performance: {overall_performance}\n\n"
        f"Your task:\n"
        f"- Give 2 to 4 sentences of specific feedback only about eye contact, facial expressions, visible confidence, and overall engagement.\n"
        f"- Be concrete and natural, not robotic.\n"
        f"- If scores are weak, explain how to improve.\n"
        f"- If some areas are decent, mention that too.\n"
        f"- Do not mention posture or interview content.\n"
    )