import csv
import os
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

NOSE_LANDMARK_INDEX = 0
LEFT_SHOULDER_LANDMARK_INDEX = 11
RIGHT_SHOULDER_LANDMARK_INDEX = 12

DEFAULT_POSE_LANDMARKER_PATH = os.path.join(
    os.path.dirname(__file__),
    "pose_landmarker.task",
)
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "posture_model.h5")
DEFAULT_SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.save")
RECRUITVIEW_CONFIDENCE_ALIASES = (
    "confidence_score",
    "confidence",
    "Confidence Score",
    "Confidence",
)
RECRUITVIEW_FACIAL_ALIASES = (
    "facial_expression",
    "facial_expression_score",
    "facial expression",
    "Facial Expression",
)


def extract_features(landmarks):
    left_shoulder = np.array(
        [
            landmarks[LEFT_SHOULDER_LANDMARK_INDEX].x,
            landmarks[LEFT_SHOULDER_LANDMARK_INDEX].y,
        ]
    )
    right_shoulder = np.array(
        [
            landmarks[RIGHT_SHOULDER_LANDMARK_INDEX].x,
            landmarks[RIGHT_SHOULDER_LANDMARK_INDEX].y,
        ]
    )
    nose = np.array(
        [
            landmarks[NOSE_LANDMARK_INDEX].x,
            landmarks[NOSE_LANDMARK_INDEX].y,
        ]
    )

    neck_base = (left_shoulder + right_shoulder) / 2
    dx = nose[0] - neck_base[0]
    dy = nose[1] - neck_base[1]
    head_tilt_angle = float(np.degrees(np.arctan2(dy, dx)))

    v1 = left_shoulder - neck_base
    v2 = right_shoulder - neck_base
    shoulder_norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if shoulder_norm_product == 0:
        return None

    cos_angle = np.dot(v1, v2) / shoulder_norm_product
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    shoulder_angle = float(np.degrees(np.arccos(cos_angle)))

    left_z = landmarks[LEFT_SHOULDER_LANDMARK_INDEX].z
    right_z = landmarks[RIGHT_SHOULDER_LANDMARK_INDEX].z
    nose_z = landmarks[NOSE_LANDMARK_INDEX].z
    z_diff = float(((left_z + right_z) / 2) - nose_z)

    return head_tilt_angle, shoulder_angle, z_diff


def create_pose_landmarker(model_path=DEFAULT_POSE_LANDMARKER_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Missing MediaPipe pose model. Download `pose_landmarker.task` "
            f"and place it at '{model_path}', or pass model_path explicitly."
        )

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect_video_pose_landmarks(pose_detector, frame, frame_index, fps):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int((frame_index / fps) * 1000)
    result = pose_detector.detect_for_video(mp_image, timestamp_ms)

    if result.pose_landmarks:
        return result.pose_landmarks[0]

    return None


def get_feedback(score, metric_name=""):
    level = int(score * 10)

    if level <= 1:
        return f"{metric_name}Severe issue detected. Needs immediate correction."
    if level <= 3:
        return f"{metric_name}Poor posture. Focus on improving this area."
    if level <= 5:
        return f"{metric_name}Needs improvement."
    if level <= 7:
        return f"{metric_name}Doing okay, but there's room for improvement."
    if level <= 8:
        return f"{metric_name}Good posture. Stay consistent."
    if level <= 9:
        return f"{metric_name}Great job! Very stable."
    return f"{metric_name}Excellent! Keep up the perfect posture."


def _first_present_value(sample: dict[str, Any], aliases: tuple[str, ...], field_name: str):
    for alias in aliases:
        if alias in sample:
            return float(sample[alias]), alias
    raise KeyError(
        f"Could not find `{field_name}` in RecruitView sample. Tried aliases: {aliases}"
    )


def compute_recruitview_weighted_label(
    sample: dict[str, Any],
    confidence_weight: float = 0.8,
    facial_expression_weight: float = 0.2,
) -> dict[str, Any]:
    confidence_score, confidence_key = _first_present_value(
        sample,
        RECRUITVIEW_CONFIDENCE_ALIASES,
        "confidence score",
    )
    facial_expression_score, facial_key = _first_present_value(
        sample,
        RECRUITVIEW_FACIAL_ALIASES,
        "facial expression score",
    )
    official_label = (
        confidence_weight * confidence_score
        + facial_expression_weight * facial_expression_score
    )
    return {
        "official_label": float(official_label),
        "confidence_score": confidence_score,
        "facial_expression_score": facial_expression_score,
        "confidence_key": confidence_key,
        "facial_expression_key": facial_key,
    }


def recruitview_raw_label_to_unit_interval(raw_label: float) -> float:
    return float(1.0 / (1.0 + np.exp(-raw_label)))


def _normalize_metric(value: float, good_center: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    distance = abs(value - good_center)
    return max(0.0, 1.0 - (distance / tolerance))


def _heuristic_posture_score(head_tilt: float, shoulder_angle: float, z_diff: float) -> float:
    head_score = _normalize_metric(head_tilt, good_center=-90.0, tolerance=30.0)
    shoulder_score = _normalize_metric(shoulder_angle, good_center=180.0, tolerance=35.0)
    lean_score = _normalize_metric(z_diff, good_center=0.0, tolerance=0.18)
    return float(np.clip((head_score + shoulder_score + lean_score) / 3.0, 0.0, 1.0))


def _load_optional_model(model_path: str, scaler_path: str):
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None, None

    import joblib
    import tensorflow as tf

    # Inference does not need the serialized training configuration, and
    # loading legacy H5 files with compile=True can fail across Keras versions.
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler


def _recruitview_video_path(video_value: Any) -> str:
    if isinstance(video_value, dict):
        video_path = video_value.get("path")
        if video_path:
            return str(video_path)
    if isinstance(video_value, str):
        return video_value
    raise ValueError(
        "RecruitView sample does not expose a usable video path. "
        "Load the dataset with video decoding disabled."
    )


def build_recruitview_posture_dataset(
    output_csv: str = "recruitview_posture_dataset.csv",
    split: str = "train",
    max_samples: int | None = None,
    frame_stride: int = 15,
    model_path: str = DEFAULT_POSE_LANDMARKER_PATH,
):
    from datasets import Video, load_dataset

    dataset = load_dataset("AI4A-lab/RecruitView")
    dataset = dataset.cast_column("video", Video(decode=False))
    samples = dataset[split]
    rows_written = 0
    samples_processed = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "head_tilt",
                "shoulder_angle",
                "z_diff",
                "official_label",
                "official_label_unit_interval",
                "confidence_score",
                "facial_expression_score",
                "sample_id",
                "file_name",
                "user_no",
                "question_id",
                "frame_index",
            ]
        )

        for sample in samples:
            if max_samples is not None and samples_processed >= max_samples:
                break

            label_info = compute_recruitview_weighted_label(sample)
            official_label = label_info["official_label"]
            video_path = _recruitview_video_path(sample["video"])
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"WARNING: Could not open RecruitView video, skipping: {video_path}")
                samples_processed += 1
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0

            # Each clip is its own video stream, so we create a fresh
            # landmarker per sample and let timestamps restart from zero.
            with create_pose_landmarker(model_path=model_path) as pose_detector:
                frame_index = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_index % max(frame_stride, 1) != 0:
                        frame_index += 1
                        continue

                    landmarks = detect_video_pose_landmarks(
                        pose_detector,
                        frame,
                        frame_index,
                        fps,
                    )
                    if landmarks is None:
                        frame_index += 1
                        continue

                    features = extract_features(landmarks)
                    if features is None:
                        frame_index += 1
                        continue

                    head_tilt, shoulder_angle, z_diff = features
                    writer.writerow(
                        [
                            head_tilt,
                            shoulder_angle,
                            z_diff,
                            official_label,
                            recruitview_raw_label_to_unit_interval(official_label),
                            label_info["confidence_score"],
                            label_info["facial_expression_score"],
                            sample.get("id", ""),
                            sample.get("file_name", ""),
                            sample.get("user_no", ""),
                            sample.get("question_id", ""),
                            frame_index,
                        ]
                    )
                    rows_written += 1
                    frame_index += 1

            cap.release()

            samples_processed += 1

    print(
        f"RecruitView posture dataset built with {rows_written} rows from "
        f"{samples_processed} video samples and saved to '{output_csv}'."
    )


def analyze_posture_video(
    video_path: str,
    pose_landmarker_path: str = DEFAULT_POSE_LANDMARKER_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
    frame_stride: int = 5,
) -> dict[str, Any]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    model, scaler = _load_optional_model(model_path, scaler_path)
    total_score = 0.0
    frame_count = 0
    detected_frames = 0
    sampled_features = []

    with create_pose_landmarker(model_path=pose_landmarker_path) as pose_detector:
        frame_index = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % max(frame_stride, 1) != 0:
                frame_index += 1
                continue

            landmarks = detect_video_pose_landmarks(
                pose_detector,
                frame,
                frame_index,
                fps,
            )
            frame_count += 1

            if landmarks is None:
                frame_index += 1
                continue

            features = extract_features(landmarks)
            if features is None:
                frame_index += 1
                continue

            head_tilt, shoulder_angle, z_diff = features
            sampled_features.append(
                {
                    "head_tilt": head_tilt,
                    "shoulder_angle": shoulder_angle,
                    "z_diff": z_diff,
                }
            )

            if model is not None and scaler is not None:
                features_array = np.array([[head_tilt, shoulder_angle, z_diff]])
                features_scaled = scaler.transform(features_array)
                posture_score = float(model.predict(features_scaled, verbose=0)[0][0])
            else:
                posture_score = _heuristic_posture_score(
                    head_tilt,
                    shoulder_angle,
                    z_diff,
                )

            total_score += posture_score
            detected_frames += 1
            frame_index += 1

    cap.release()

    if detected_frames == 0:
        return {
            "score": 0.0,
            "feedback": "No valid posture frames were detected. Make sure your upper body is visible and the video has enough lighting.",
            "frames_analyzed": frame_count,
            "detected_frames": detected_frames,
            "used_trained_model": model is not None and scaler is not None,
            "metrics": {},
        }

    final_score = total_score / detected_frames
    display_score = (
        recruitview_raw_label_to_unit_interval(final_score)
        if model is not None and scaler is not None
        else final_score
    )
    metrics = {
        "average_head_tilt": float(
            np.mean([feature["head_tilt"] for feature in sampled_features])
        ),
        "average_shoulder_angle": float(
            np.mean([feature["shoulder_angle"] for feature in sampled_features])
        ),
        "average_forward_lean": float(
            np.mean([feature["z_diff"] for feature in sampled_features])
        ),
    }

    feedback = get_feedback(display_score, "")
    if metrics["average_forward_lean"] < -0.08:
        feedback += " You appear to lean forward quite a bit, so try sitting back and keeping your neck stacked over your shoulders."
    if metrics["average_shoulder_angle"] < 150:
        feedback += " Your shoulders also look somewhat closed in, so relaxing them down and back should help."

    return {
        "score": round(display_score, 4),
        "raw_score": round(final_score, 4),
        "feedback": feedback,
        "frames_analyzed": frame_count,
        "detected_frames": detected_frames,
        "used_trained_model": model is not None and scaler is not None,
        "metrics": metrics,
    }


def train_model(
    csv_path="recruitview_posture_dataset.csv",
    model_out="posture_model.h5",
    scaler_out="scaler.save",
):
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from tensorflow import keras

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from '{csv_path}'")

    expected_columns = {"head_tilt", "shoulder_angle", "z_diff", "official_label"}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns for RecruitView posture training: {sorted(missing_columns)}"
        )

    X = df[["head_tilt", "shoulder_angle", "z_diff"]].values
    y = df["official_label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(3,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="linear"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    model.summary()

    model.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest loss: {loss:.4f}")
    print(f"Test MAE : {mae:.4f}")

    model.save(model_out)
    joblib.dump(scaler, scaler_out)
    print(f"Model saved to '{model_out}', scaler saved to '{scaler_out}'")
