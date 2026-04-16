import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from datasets import load_dataset

# added this to import the dataset that we plan to use
ds = load_dataset("AI4A-lab/RecruitView")

NOSE_LANDMARK_INDEX = 0
LEFT_SHOULDER_LANDMARK_INDEX = 11
RIGHT_SHOULDER_LANDMARK_INDEX = 12
DEFAULT_POSE_LANDMARKER_PATH = os.path.join(
    os.path.dirname(__file__),
    "pose_landmarker.task",
)



def extract_features(landmarks):
    """
    Takes MediaPipe landmark data from one frame and returns 3 posture features:
      1. head_tilt_angle  — how far the head tilts left/right (degrees)
      2. shoulder_angle   — angle between shoulders; smaller = more scrunched
      3. z_diff           — forward lean: difference in depth between shoulders and nose

    These 3 numbers become the INPUT to the neural network.
    """
 
    left_shoulder = np.array([
        landmarks[LEFT_SHOULDER_LANDMARK_INDEX].x,
        landmarks[LEFT_SHOULDER_LANDMARK_INDEX].y,
    ])
    right_shoulder = np.array([
        landmarks[RIGHT_SHOULDER_LANDMARK_INDEX].x,
        landmarks[RIGHT_SHOULDER_LANDMARK_INDEX].y,
    ])
    nose = np.array([
        landmarks[NOSE_LANDMARK_INDEX].x,
        landmarks[NOSE_LANDMARK_INDEX].y,
    ])

    # head tilt calc
    neck_base = (left_shoulder + right_shoulder) / 2
    dx = nose[0] - neck_base[0]
    dy = nose[1] - neck_base[1]
    head_tilt_angle = np.degrees(np.arctan2(dy, dx))

    # shoulder calc
    v1 = left_shoulder  - neck_base
    v2 = right_shoulder - neck_base
    shoulder_norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if shoulder_norm_product == 0:
        return None

    cos_angle = np.dot(v1, v2) / shoulder_norm_product
    cos_angle = np.clip(cos_angle, -1.0, 1.0)   # prevent arccos domain errors
    shoulder_angle = np.degrees(np.arccos(cos_angle))

    # leaning calc
    left_z = landmarks[LEFT_SHOULDER_LANDMARK_INDEX].z
    right_z = landmarks[RIGHT_SHOULDER_LANDMARK_INDEX].z
    nose_z = landmarks[NOSE_LANDMARK_INDEX].z
    z_diff  = ((left_z + right_z) / 2) - nose_z

    return head_tilt_angle, shoulder_angle, z_diff


def create_pose_landmarker(model_path=DEFAULT_POSE_LANDMARKER_PATH):
    """
    Creates the supported MediaPipe Tasks pose detector for video frames.

    We require the `.task` model file explicitly because the legacy
    Solutions API is no longer available in newer MediaPipe releases.
    """
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
    """
    Runs pose detection on one video frame and returns the first detected pose.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((frame_index / fps) * 1000)
    result = pose_detector.detect_for_video(mp_image, timestamp_ms)

    if result.pose_landmarks:
        return result.pose_landmarks[0]

    return None


def draw_pose_landmarks(frame, landmarks):
    """
    Draws simple landmark points with OpenCV so inference stays visual even
    after moving away from the removed Solutions drawing helpers.
    """
    frame_height, frame_width = frame.shape[:2]

    for landmark in landmarks:
        x_coord = int(landmark.x * frame_width)
        y_coord = int(landmark.y * frame_height)
        cv2.circle(frame, (x_coord, y_coord), 3, (255, 255, 0), -1)


#just general feedback
def get_feedback(score, metric_name="Overall Posture"):
    """
    Converts the neural network's output (a float 0–1) into human-readable text.
    Higher score = better posture.
    """
    level = int(score * 10)
    if level <= 1:
        return f"{metric_name}: Severe issue detected. Needs immediate correction."
    elif level <= 3:
        return f"{metric_name}: Poor posture. Focus on improving this area."
    elif level <= 5:
        return f"{metric_name}: Needs improvement."
    elif level <= 7:
        return f"{metric_name}: Doing okay, but there's room for improvement."
    elif level <= 8:
        return f"{metric_name}: Good posture. Stay consistent."
    elif level <= 9:
        return f"{metric_name}: Great job! Very stable."
    else:
        return f"{metric_name}: Excellent! Keep up the perfect posture."


# =============================================================
# STAGE 1 — DATASET BUILDER
# =============================================================
# PURPOSE:
#   Loop through a folder of labeled videos, extract 3 posture
#   features per frame, and save everything to a CSV file.
#
# HOW TO USE:
#   - Put your "good posture" videos in one folder, "bad" in another.
#   - Set the paths and labels below, then call build_dataset().
#
# OUTPUT:
#   posture_dataset.csv  (columns: head_tilt, shoulder_angle, z_diff, label)
#   label = 1 means GOOD posture, label = 0 means BAD posture.
# =============================================================

def build_dataset(video_sources,
                  output_csv="posture_dataset.csv",
                  model_path=DEFAULT_POSE_LANDMARKER_PATH):
    """
    video_sources: list of (video_path, label) tuples
        e.g. [("good1.avi", 1), ("bad1.avi", 0), ...]
    output_csv: where to save the collected features
    """
    import csv

    rows_written = 0

    # Open CSV in append mode so you can add more videos later
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)

        for video_path, label in video_sources:
            print(f"Processing: {video_path}  (label={label})")
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"  WARNING: Could not open {video_path}, skipping.")
                continue

            # Create a fresh detector per video so video timestamps can restart
            # from zero without violating MediaPipe's monotonic timestamp rule.
            with create_pose_landmarker(model_path=model_path) as pose_detector:
                frame_index = 0
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    landmarks = detect_video_pose_landmarks(
                        pose_detector,
                        frame,
                        frame_index,
                        fps,
                    )
                    if landmarks:
                        features = extract_features(landmarks)
                        if features is not None:
                            head_tilt, shoulder_angle, z_diff = features
                            writer.writerow([head_tilt, shoulder_angle, z_diff, label])
                            rows_written += 1

                    frame_index += 1

            cap.release()

    print(f"\nDataset built! {rows_written} frames saved to '{output_csv}'")


# ── Example call for Stage 1 ──────────────────────────────────
# change the paths to the correct binary label.
#
video_sources = [
     (r"/Users/rahulganesh/Desktop/USC/CSE 566/Project/CSCI-566-Course-Project-DeepPrep-AI/backend/Posture/videos/P22.avi", 1),  # good posture
     (r"/Users/rahulganesh/Desktop/USC/CSE 566/Project/CSCI-566-Course-Project-DeepPrep-AI/backend/Posture/videos/P24.avi", 0),  # bad posture
     (r"/Users/rahulganesh/Desktop/USC/CSE 566/Project/CSCI-566-Course-Project-DeepPrep-AI/backend/Posture/videos/P25.avi", 1),
     (r"/Users/rahulganesh/Desktop/USC/CSE 566/Project/CSCI-566-Course-Project-DeepPrep-AI/backend/Posture/videos/P27.avi", 1),
     (r"/Users/rahulganesh/Desktop/USC/CSE 566/Project/CSCI-566-Course-Project-DeepPrep-AI/backend/Posture/videos/P29.avi", 1),
     (r"/Users/rahulganesh/Desktop/USC/CSE 566/Project/CSCI-566-Course-Project-DeepPrep-AI/backend/Posture/videos/P30.avi", 0),
     # ... add as many as you have
]


# =============================================================
# STAGE 2 — NEURAL NETWORK TRAINING
# =============================================================
# PURPOSE:
#   Load the CSV built in Stage 1, normalize the features,
#   train a small neural network, and save it to disk.
#
# WHAT THE NETWORK LOOKS LIKE:
#   Input  (3 features)
#     → Dense layer 64 neurons, ReLU activation   ← learns patterns
#     → Dropout 30%                                ← prevents overfitting
#     → Dense layer 32 neurons, ReLU activation   ← refines patterns
#     → Dropout 30%
#     → Dense layer 1 neuron,  Sigmoid activation ← outputs 0–1 score
#
# OUTPUT:
#   posture_model.h5   (the trained neural network)
#   scaler.save        (the normalization object — needed at inference time)
# =============================================================

def train_model(csv_path="posture_dataset.csv",
                model_out="posture_model.h5",
                scaler_out="scaler.save"):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import joblib
    import tensorflow as tf
    from tensorflow import keras

    # -- Load data --
    df = pd.read_csv(csv_path, header=None,
                     names=["head_tilt", "shoulder_angle", "z_diff", "label"])
    print(f"Loaded {len(df)} rows from '{csv_path}'")

    X = df[["head_tilt", "shoulder_angle", "z_diff"]].values
    y = df["label"].values

    # -- Normalize features --
    # Neural networks train much better when all inputs are on a similar scale.
    # StandardScaler shifts each feature to mean=0, std=1.
    # We save the scaler so Stage 3 can apply the SAME transformation.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -- Train / test split --
    # 80% of frames go to training, 20% held back to test accuracy.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # -- Build neural network --
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(3,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # binary_crossentropy is the standard loss for 0/1 classification
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # -- Train --
    model.fit(X_train, y_train,
              epochs=50,
              batch_size=32,
              validation_data=(X_test, y_test))

    # -- Evaluate on test set --
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.2%}")

    # -- Save --
    model.save(model_out)
    joblib.dump(scaler, scaler_out)
    print(f"Model saved to '{model_out}', scaler saved to '{scaler_out}'")


# ── Example call for Stage 2 ──────────────────────────────────
# Uncomment after Stage 1 is complete.
#
# train_model()


# =============================================================
# STAGE 3 — REAL-TIME INFERENCE
# =============================================================
# PURPOSE:
#   Load the saved model and scaler, run MediaPipe on a new video,
#   pass each frame's features through the network, display the
#   live score on screen, and print a final summary at the end.
#
# WHAT CHANGES vs. YOUR ORIGINAL CODE:
#   - Scoring is done by model.predict() instead of manual math
#   - The score is the network's output (0–1), not a hand-tuned formula
#   - Everything else (MediaPipe, landmark drawing, feedback) stays the same
# =============================================================

def run_inference(video_path,
                  model_path="posture_model.h5",
                  scaler_path="scaler.save",
                  pose_landmarker_path=DEFAULT_POSE_LANDMARKER_PATH):
    import joblib
    import tensorflow as tf

    # -- Load model and scaler from disk --
    model  = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_score = 0.0
    frame_count = 0

    with create_pose_landmarker(model_path=pose_landmarker_path) as pose_detector:
        frame_index = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = detect_video_pose_landmarks(
                pose_detector,
                frame,
                frame_index,
                fps,
            )

            if landmarks:
                # Step 1: Extract the same 3 features as in Stage 1
                features = extract_features(landmarks)
                if features is not None:
                    head_tilt, shoulder_angle, z_diff = features

                    # Step 2: Normalize using the saved scaler
                    features_array = np.array([[head_tilt, shoulder_angle, z_diff]])
                    features_scaled = scaler.transform(features_array)

                    # Step 3: Get posture score from the neural network
                    posture_score = float(
                        model.predict(features_scaled, verbose=0)[0][0]
                    )

                    total_score += posture_score
                    frame_count += 1

                    # Draw landmarks and overlay score on the frame
                    draw_pose_landmarks(frame, landmarks)
                    color = (0, 255, 0) if posture_score >= 0.5 else (0, 0, 255)
                    cv2.putText(frame, f"Posture: {posture_score:.2f}",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

            cv2.imshow("Posture Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    # -- Print final summary --
    if frame_count > 0:
        final_score = total_score / frame_count
        feedback = get_feedback(final_score, "Overall Posture")
        print("\n========== POSTURE ANALYSIS SUMMARY ==========")
        print(f"Frames analysed : {frame_count}")
        print(f"Average score   : {final_score:.2f}  (0=bad, 1=good)")
        print(f"Feedback        : {feedback}")
        print("===============================================")


# ── Example call for Stage 3 ──────────────────────────────────
# Uncomment after Stage 2 is complete.
#
# run_inference(r"C:\Users\Sherwin\Documents\Training-posture\P21.avi")


# =============================================================
# MAIN — change the STAGE variable to control what runs
# =============================================================
if __name__ == "__main__":

    STAGE = 3   # ← set to 1, 2, or 3

    if STAGE == 1:
        build_dataset(video_sources)

    elif STAGE == 2:
        train_model()

    elif STAGE == 3:
        run_inference(r"/Users/rahulganesh/Desktop/USC/CSE 566/Project/CSCI-566-Course-Project-DeepPrep-AI/backend/Posture/videos/P31.avi")
