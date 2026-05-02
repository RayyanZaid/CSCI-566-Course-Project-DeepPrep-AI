# Run Neural Network on a local video path (end-to-end).

import os
import tempfile
import numpy as np
import torch
import librosa
from moviepy import VideoFileClip
from sentence_transformers import SentenceTransformer
from transformers import pipeline
try:
    from .NN import MultimodalTransformerRegressor
except ImportError:
    from NN import MultimodalTransformerRegressor
MAX_SEQ_LEN = 64
embed_dim = 384
PROSODY_DIM = 13

LABEL_COLS = [
    "interview_score",
    "overall_personality",
    "answer_score",
    "speaking_skills",
    "agreeableness",
    "conscientiousness",
    "neuroticism",
    "openness",
    "confidence_score",
]

def evaluateAudio(videoPath: str):
    
    # ----------------------------
    # 1) Transcribe local video -> timestamped transcript segments
    # ----------------------------


    # transcript_segments = transcribe_video_to_transcript_segments(videoPath, asr_model="openai/whisper-small")

    # TODO: Remove hardcoded and uncomment above line
    transcript_segments = {
    "0:00 - 0:11": "I am a third year BTEC student from Manipal University, Jaipur currently majoring in CS,",
    "0:11 - 0:13": "computer science.",
    "0:13 - 0:18": "I am a tech enthusiast and I love to explore the domains of artificial intelligence and",
    "0:18 - 0:19": "machine learning.",
    "0:19 - 0:26": "I also have a sure interest in exploring the realm of blockchain and its decentralized",
    "0:26 - 0:34": "apps with an interest of playing with softwares. My hobbies include dancing, reading, I am an",
    "0:34 - 0:42": "avid reader and also exploring new places, meeting new people, their culture and their",
    "0:42 - 0:42": "heritage.",
    }


    segment_texts = list(transcript_segments.values())


    # ----------------------------
    # 1) Build segment-level text + prosody features
    # ----------------------------

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    text_embeddings = st_model.encode(segment_texts).astype(np.float32)  # (n_segments, embed_dim)
    prosody_features = parse_audio_return_prosody_features(videoPath, transcript_segments)  # (n_segments, 13)

    # Keep aligned segment count across modalities
    n = min(text_embeddings.shape[0], prosody_features.shape[0], MAX_SEQ_LEN)
    if n == 0:
        raise ValueError("No valid segments after feature extraction.")

    x_text = np.zeros((1, MAX_SEQ_LEN, embed_dim), dtype=np.float32)
    x_prosody = np.zeros((1, MAX_SEQ_LEN, PROSODY_DIM), dtype=np.float32)
    x_mask = np.ones((1, MAX_SEQ_LEN), dtype=bool)

    x_text[0, :n, :] = text_embeddings[:n, :embed_dim]
    x_prosody[0, :n, :] = prosody_features[:n, :]
    x_mask[0, :n] = False


    # ----------------------------
    # 2) Predict with trained model
    # ----------------------------

    # Get model from path: multimodal_transformer_audio_regressor.pth

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalTransformerRegressor(
        text_dim=embed_dim,
        prosody_dim=PROSODY_DIM,
        max_seq_len=MAX_SEQ_LEN,
        num_targets=len(LABEL_COLS),
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.25,
        fusion="concat",
    ).to(device)
    model_path = os.path.join(os.path.dirname(__file__), "multimodal_transformer_audio_regressor.pth") # saved this model after training in NN.ipynb
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)


    model.eval()
    with torch.no_grad():
        pred = model(
            torch.tensor(x_text, dtype=torch.float32, device=device),
            torch.tensor(x_mask, dtype=torch.bool, device=device),
            torch.tensor(x_prosody, dtype=torch.float32, device=device),
        )
    pred = pred.squeeze(0).detach().cpu().numpy()

    prediction_dict = {label: float(value) for label, value in zip(LABEL_COLS, pred)}
    print("\nPredicted scores:")
    for k, v in prediction_dict.items():
        print(f"  {k:22s}: {v:.4f}")

    print("\nTranscript segments used by the model:")
    for i, (k, v) in enumerate(transcript_segments.items()):
        print(f"  [{k}] {v}")
    
    audio_interview_score = prediction_dict["interview_score"]
    audio_overall_personality = prediction_dict["overall_personality"]
    audio_answer_score = prediction_dict["answer_score"]
    audio_speaking_skills = prediction_dict["speaking_skills"]
    audio_agreeableness = prediction_dict["agreeableness"]
    audio_conscientiousness = prediction_dict["conscientiousness"]  
    audio_neuroticism = prediction_dict["neuroticism"]
    audio_openness = prediction_dict["openness"]
    audio_confidence_score = prediction_dict["confidence_score"]

    return audio_interview_score, audio_overall_personality, audio_answer_score, audio_speaking_skills, audio_agreeableness, audio_conscientiousness, audio_neuroticism, audio_openness, audio_confidence_score





def getInterviewResponseContentFeedback(audio_interview_score, audio_overall_personality, audio_answer_score, audio_speaking_skills, audio_agreeableness, audio_conscientiousness, audio_neuroticism, audio_openness, audio_confidence_score):
    interview_response_content_feedback = f"""
    You are an expert interview coach providing personalized, encouraging, and actionable feedback 
    to a job candidate based on their recorded audio interview scores.

    All scores are standardized (z-scores), where 0 = average, negative = below average, 
    positive = above average. Use the thresholds below to interpret each score.

    --- CANDIDATE SCORES ---
    audio_interview_score: {audio_interview_score}
    audio_overall_personality: {audio_overall_personality}
    audio_answer_score: {audio_answer_score}
    audio_speaking_skills: {audio_speaking_skills}
    audio_agreeableness: {audio_agreeableness}
    audio_conscientiousness: {audio_conscientiousness}
    audio_neuroticism: {audio_neuroticism}
    audio_openness: {audio_openness}
    audio_confidence_score: {audio_confidence_score}

    --- SCORE INTERPRETATION GUIDE ---

    INTERVIEW SCORE (audio_interview_score) — overall composite performance:
    - High (> 0.55): Excellent interview. You performed in the top 25% of all candidates. Your answers were compelling and well-delivered.
    - Mid (-0.60 to 0.55): Solid interview. There are specific areas to sharpen that could push you into top-performer territory.
    - Low (< -0.60): This interview needs significant improvement. Focus on the specific feedback below to rebuild your foundation.

    OVERALL PERSONALITY (audio_overall_personality) — holistic impression from audio:
    - High (> 0.62): Your personality came through strongly and positively. You left a memorable impression.
    - Mid (-0.61 to 0.62): Your personality signals are average. Work on letting more of your authentic self come through vocally.
    - Low (< -0.61): Your audio personality signal was muted. Interviewers may have found you hard to read or connect with.

    ANSWER SCORE (audio_answer_score) — quality and relevance of spoken answers:
    - High (> 0.58): Your answers were clear, relevant, and compelling. You addressed questions effectively.
    - Mid (-0.61 to 0.58): Your answers were generally on track but could be more structured or specific. Try the STAR method (Situation, Task, Action, Result).
    - Low (< -0.61): Your answers may have been off-topic, too vague, or incomplete. Practice structuring responses before your next interview.

    SPEAKING SKILLS (audio_speaking_skills) — clarity, pace, and articulation:
    - High (> 0.62): Outstanding delivery. You spoke clearly, at a good pace, and were easy to follow.
    - Mid (-0.56 to 0.62): Your speaking is average. Consider recording yourself to identify filler words, pacing issues, or mumbling.
    - Low (< -0.56): Speaking clarity needs work. Practice enunciation, slow your pace, and reduce filler words like "um" and "uh."

    AGREEABLENESS (audio_agreeableness) — warmth, cooperation, and social tone:
    - High (> 0.55): You came across as warm, collaborative, and easy to work with — a strong interpersonal signal.
    - Mid (-0.57 to 0.55): Your agreeableness was neutral. Try to express more enthusiasm and positivity in your tone.
    - Low (< -0.57): You may have come across as cold, guarded, or combative. Focus on using warmer language and an open, friendly tone.

    CONSCIENTIOUSNESS (audio_conscientiousness) — reliability and diligence signals:
    - High (> 0.49): You conveyed strong dependability and attention to detail — employers love this.
    - Mid (-0.45 to 0.49): Most candidates score here. To stand out, add specific examples of follow-through and reliability in your answers.
    - Low (< -0.45): You may have seemed disorganized or unprepared. Structure your answers more carefully and reference concrete examples.

    NEUROTICISM (audio_neuroticism) — anxiety and emotional stability signals:
    NOTE: This score has a very tight range across candidates. High scores mean MORE anxious-sounding; low means calmer.
    - Low (< -0.31): You sounded emotionally calm and stable — a strong positive signal.
    - Mid (-0.31 to 0.30): Normal range. Most candidates sound mildly nervous in interviews — this is expected.
    - High (> 0.30): You sounded noticeably anxious or tense. Practice deep breathing before interviews and try mock interview sessions to build comfort.

    OPENNESS (audio_openness) — intellectual curiosity and engagement signals:
    - High (> 0.52): You conveyed genuine curiosity and enthusiasm. You sounded engaged and eager to learn.
    - Mid (-0.55 to 0.52): Average openness. Try asking more thoughtful questions and expressing curiosity about the role and company.
    - Low (< -0.55): You may have seemed disinterested or rigid. Show enthusiasm by referencing specific things that excite you about the opportunity.

    CONFIDENCE SCORE (audio_confidence_score) — assertiveness and self-assurance:
    - High (> 0.55): Excellent. You projected confidence and authority without being arrogant.
    - Mid (-0.54 to 0.55): Moderate confidence. Work on speaking more assertively — avoid upward inflections that make statements sound like questions.
    - Low (< -0.54): Low confidence signals can hurt your candidacy. Practice power poses, record your voice, and rehearse your answers until they feel natural.

    --- YOUR TASK ---

    Note: For all scores - Give them the numbers and score them on a scale of 1 to 10 (10 being top-tier). Like convert the (0.55, 0.62, -0.60, etc) thresholds into a 1-10 scale and give them that score in your feedback so they know where they stand.

    1. Start with a 2-3 sentence overall summary of this candidate's interview performance based 
    on their interview_score and overall_personality. 
    2. Identify their TOP 2 strengths (highest scoring areas) and give specific, encouraging praise.

    3. Identify their TOP 2 areas for improvement (lowest scoring areas) and give concrete, 
    actionable advice — not just what to improve, but HOW. Be direct but kind.

    Keep the tone warm, expert, and direct. Do not be generic. Reference the actual scores in 
    your feedback so the candidate knows exactly where they stand.
        """
    
    return interview_response_content_feedback








# IGNORE BELOW

# ----------------------------
# Helpers reused from Audio_Embedding.ipynb
# ----------------------------
def parse_timestamp_range(ts_string):
    start_str, end_str = ts_string.split(" - ")

    def to_seconds(t):
        parts = t.split(":")
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        raise ValueError(f"Invalid timestamp format: {t}")

    return (to_seconds(start_str), to_seconds(end_str))

def seconds_to_mmss(seconds):
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = int(round(seconds - 60 * m))
    if s == 60:
        m += 1
        s = 0
    return f"{m}:{s:02d}"

def extract_energy(speech):
    return librosa.feature.rms(y=speech)[0]

def extract_pitch(speech, rate):
    pitch, _ = librosa.core.piptrack(y=speech, sr=rate)
    pitch = pitch[pitch > 0]
    return pitch

def pitch_stats(pitch):
    if pitch is None or len(pitch) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(np.min(pitch)), float(np.max(pitch)), float(np.mean(pitch)), float(np.std(pitch)))

def intensity_features(speech):
    intensity = librosa.feature.rms(y=speech)[0]
    return (float(np.min(intensity)), float(np.max(intensity)), float(np.mean(intensity)))

def jitter_shimmer(speech, rate):
    pitch, _ = librosa.core.piptrack(y=speech, sr=rate)
    voiced_pitch = pitch[pitch > 0]
    jitter = float(np.std(voiced_pitch)) if voiced_pitch.size > 0 else 0.0
    intensity = librosa.feature.rms(y=speech)[0]
    shimmer = float(np.std(intensity)) if intensity.size > 0 else 0.0
    return jitter, shimmer

def speech_rate(speech, rate):
    onset_env = librosa.onset.onset_strength(y=speech, sr=rate)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=rate)
    duration = len(speech) / rate if rate > 0 else 0.0
    if duration <= 0:
        return 0.0
    return float(len(onset_frames) / duration)

def pause_features(speech, rate):
    onset_env = librosa.onset.onset_strength(y=speech, sr=rate)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=rate)
    intervals = librosa.frames_to_time(onset_frames, sr=rate)
    pauses = np.diff(intervals)
    max_pause = float(np.max(pauses)) if len(pauses) > 0 else 0.0
    avg_pause = float(np.mean(pauses)) if len(pauses) > 0 else 0.0
    return max_pause, avg_pause

def analyze_prosody(video_path, timestamp_ranges):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
        audio_path = tmp_audio_file.name

    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        return [np.zeros((1, 13), dtype=np.float32) for _ in timestamp_ranges]

    clip.audio.write_audiofile(audio_path, logger=None)
    clip.close()

    speech, rate = librosa.load(audio_path, sr=None)
    features = []

    for start, end in timestamp_ranges:
        start_sample = int(start * rate)
        end_sample = int(end * rate)
        segment_speech = speech[start_sample:end_sample]

        if len(segment_speech) == 0:
            features.append(np.zeros((1, 13), dtype=np.float32))
            continue

        energy = extract_energy(segment_speech)
        min_pitch, max_pitch, mean_pitch, pitch_sd = pitch_stats(extract_pitch(segment_speech, rate))
        intensity_min, intensity_max, intensity_mean = intensity_features(segment_speech)
        jitter, shimmer = jitter_shimmer(segment_speech, rate)
        speak_rate = speech_rate(segment_speech, rate)
        max_pause, avg_pause = pause_features(segment_speech, rate)

        feature_vec = np.array([
            float(np.mean(energy)) if energy.size > 0 else 0.0,
            min_pitch,
            max_pitch,
            mean_pitch,
            pitch_sd,
            intensity_min,
            intensity_max,
            intensity_mean,
            jitter,
            shimmer,
            speak_rate,
            max_pause,
            avg_pause,
        ], dtype=np.float32).reshape(1, -1)

        features.append(feature_vec)

    return features

def parse_audio_return_prosody_features(video_path, transcript_segments):
    timestamp_ranges = [parse_timestamp_range(ts) for ts in transcript_segments.keys()]
    if not timestamp_ranges:
        return np.empty((0, 13), dtype=np.float32)
    return np.vstack(analyze_prosody(video_path, timestamp_ranges)).astype(np.float32)

def transcribe_video_to_transcript_segments(video_path, asr_model="openai/whisper-small"):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
        audio_path = tmp_audio_file.name

    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        raise ValueError("Video has no audio track; cannot transcribe.")
    clip.audio.write_audiofile(audio_path, logger=None)
    clip.close()

    asr = pipeline(
        task="automatic-speech-recognition",
        model=asr_model,
        device=0 if torch.cuda.is_available() else -1,
    )

    result = asr(audio_path, return_timestamps=True)
    chunks = result.get("chunks", [])

    transcript_segments = {}
    if len(chunks) > 0:
        for c in chunks:
            ts = c.get("timestamp", None)
            text = c.get("text", "").strip()
            if not ts or len(ts) != 2 or not text:
                continue
            start, end = ts
            if start is None or end is None or end <= start:
                continue
            key = f"{seconds_to_mmss(start)} - {seconds_to_mmss(end)}"
            transcript_segments[key] = text
    else:
        full_text = result.get("text", "").strip()
        if full_text:
            duration = librosa.get_duration(path=audio_path)
            key = f"0:00 - {seconds_to_mmss(duration)}"
            transcript_segments[key] = full_text

    if len(transcript_segments) == 0:
        raise ValueError("Transcription produced no usable timestamped segments.")

    return transcript_segments


if __name__ == "__main__":
    evaluateAudio("/Users/rayyanzaid/Desktop/School/CSCI-566-DeepLearning/CSCI-566-Course-Project-DeepPrep-AI/testVideos/vid_0001.mp4")
