"""
inference.py
------------
Run a trained InterviewAnalysisNet on a new interview video file
and get a report on eye contact quality and facial expression scores.

Usage:
    python inference.py --video path/to/interview.mp4
    python inference.py --video interview.mp4 --checkpoint checkpoints/best_model.pt
    python inference.py --video interview.mp4 --report  # save JSON report
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import cv2

from backend.FacialExpressionsAndEyeContact.data_loader import FaceFeatureExtractor, VISUAL_TARGETS
from backend.FacialExpressionsAndEyeContact.model import build_model

# ── Human-readable descriptions for each target metric ───────────────────────
METRIC_DESCRIPTIONS = {
    "extraversion":            "Social engagement & expressiveness",
    "confidence":              "Confidence & presence",
    "engagement":              "Engagement & energy level",
    "professional_appearance": "Professional appearance & composure",
    "overall_performance":     "Overall interview performance",
    "openness":                "Openness to new ideas",
    "conscientiousness":       "Conscientiousness & detail-orientation",
    "agreeableness":           "Agreeableness & warmth",
    "neuroticism":             "Emotional stability (lower=more stable)",
    "overall_personality":     "Overall personality impression",
    "communication":           "Communication clarity",
    "clarity":                 "Clarity of thought & expression",
}

# Score thresholds for qualitative feedback (scores are ~centered around 0)
def score_to_grade(score: float) -> tuple:
    """Map a normalized score to (grade, feedback)."""
    if score > 0.75:
        return "Excellent", "Outstanding performance in this dimension."
    elif score > 0.40:
        return "Good", "Strong performance with minor room for improvement."
    elif score > 0.10:
        return "Average", "Meets expectations. Some areas to develop."
    elif score > -0.20:
        return "Below Average", "Needs improvement. See tips below."
    else:
        return "Poor", "Significant improvement needed in this area."

IMPROVEMENT_TIPS = {
    "confidence": [
        "Maintain steady eye contact with the camera (treat it as the interviewer's eyes).",
        "Sit upright and avoid slouching.",
        "Speak at a measured pace — rushing signals nervousness.",
    ],
    "engagement": [
        "Vary your facial expressions to match the content of your answer.",
        "Nod periodically to show active engagement.",
        "Use hand gestures moderately to emphasize key points.",
    ],
    "extraversion": [
        "Smile more naturally at the start of your answer.",
        "Project energy — a slightly louder, more animated voice helps.",
    ],
    "professional_appearance": [
        "Ensure good lighting (light source in front of you, not behind).",
        "Maintain a neutral, pleasant expression as your resting face.",
        "Minimize fidgeting or touching your face.",
    ],
    "overall_performance": [
        "Practice the STAR method (Situation, Task, Action, Result) for structured answers.",
        "Record yourself and review for any distracting habits.",
    ],
}


def load_video_frames(video_path: str, num_frames: int = 24, img_size: int = 224):
    """Load uniformly sampled frames from a local MP4 file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps if fps > 0 else 0.0

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    extractor = FaceFeatureExtractor()  # no args — img_size was removed in v2

    # ImageNet normalization — must match what the model was trained with
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    frames_list, features_list = [], []
    faces_detected = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame_bgr = cap.read()
        if not ret:
            frames_list.append(torch.zeros(3, img_size, img_size))
            features_list.append(torch.zeros(FaceFeatureExtractor.FEATURE_DIM))
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (img_size, img_size))

        # Apply ImageNet normalization (raw /255 would mismatch training)
        frame_float = (frame_resized.astype(np.float32) / 255.0 - mean) / std
        frame_tensor = torch.from_numpy(frame_float).permute(2, 0, 1)  # [3, H, W]

        feats = extractor.extract(frame_rgb)
        if feats.sum() != 0:
            faces_detected += 1

        frames_list.append(frame_tensor)
        features_list.append(feats)

    cap.release()

    frames = torch.stack(frames_list).unsqueeze(0)   # [1, T, 3, H, W]
    features = torch.stack(features_list).unsqueeze(0)  # [1, T, F]

    face_ratio = faces_detected / num_frames
    return frames, features, {"duration_sec": duration, "face_detection_rate": face_ratio}


def run_inference(
    video_path: str,
    checkpoint_path: str,
    target_cols: list = None,
    num_frames: int = 24,
    img_size: int = 224,
    device: str = "auto",
) -> dict:
    """
    Run the model on a video and return a structured report dict.

    Returns dict with keys: scores, grades, tips, metadata
    """
    # Device
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        dev = torch.device(device)

    # Load checkpoint first to read what the model was trained on
    ckpt = torch.load(checkpoint_path, map_location=dev)

    # Use target_cols and num_frames from the checkpoint unless explicitly overridden
    if target_cols is None:
        target_cols = ckpt.get("target_cols", VISUAL_TARGETS)
        if not target_cols:
            target_cols = VISUAL_TARGETS
    saved_num_frames = ckpt.get("num_frames", num_frames)
    if num_frames == 24:  # user didn't override — use what the model was trained on
        num_frames = saved_num_frames

    # Load model
    model = build_model(num_targets=len(target_cols), freeze_backbone=False).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load video
    print(f"Processing video: {video_path}")
    frames, features, meta = load_video_frames(video_path, num_frames, img_size)
    frames = frames.to(dev)
    features = features.to(dev)

    # Predict
    with torch.no_grad():
        preds = model(frames, features)  # [1, num_targets]

    scores = preds.squeeze(0).cpu().numpy()

    # Build report
    results = {}
    for col, score in zip(target_cols, scores):
        grade, feedback = score_to_grade(float(score))
        results[col] = {
            "score": round(float(score), 4),
            "grade": grade,
            "description": METRIC_DESCRIPTIONS.get(col, col),
            "feedback": feedback,
            "tips": IMPROVEMENT_TIPS.get(col, []),
        }

    # Eye contact proxy: average of gaze-related metrics
    eye_contact_cols = ["confidence", "engagement", "professional_appearance"]
    eye_scores = [results[c]["score"] for c in eye_contact_cols if c in results]
    eye_contact_score = float(np.mean(eye_scores)) if eye_scores else None

    report = {
        "video": video_path,
        "metadata": meta,
        "eye_contact_score": eye_contact_score,
        "metrics": results,
    }

    report = print(report) 
    return eye_contact_score, report


def print_report(report: dict):
    """Pretty-print the inference report to console."""
    print("\n" + "=" * 65)
    print("  INTERVIEW ANALYSIS REPORT — DeepPrep AI")
    print("=" * 65)

    meta = report["metadata"]
    print(f"  Video duration : {meta['duration_sec']:.1f}s")
    print(f"  Face detection : {meta['face_detection_rate']:.0%} of sampled frames")

    if report["eye_contact_score"] is not None:
        ec = report["eye_contact_score"]
        ec_grade, _ = score_to_grade(ec)
        print(f"\n  👁  Eye Contact Score: {ec:.3f}  [{ec_grade}]")

    print("\n  ─── Detailed Metrics ───────────────────────────────────")
    for col, info in report["metrics"].items():
        bar = "█" * max(0, int((info["score"] + 1.5) / 3.0 * 20))  # visual bar
        print(f"\n  {info['description']}")
        print(f"  Score: {info['score']:+.3f}  |  {info['grade']}")
        print(f"  [{bar:<20}]")
        print(f"  {info['feedback']}")
        if info["tips"]:
            print("  Tips:")
            for tip in info["tips"]:
                print(f"    • {tip}")

    print("\n" + "=" * 65)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to interview video (.mp4)")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--num_frames", type=int, default=24,
                        help="Frames to sample (overridden by checkpoint value if available)")
    parser.add_argument("--report", action="store_true", help="Save JSON report to disk")
    args = parser.parse_args()

    report = run_inference(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        num_frames=args.num_frames,
    )
    print_report(report)

    if args.report:
        out_path = Path(args.video).stem + "_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
