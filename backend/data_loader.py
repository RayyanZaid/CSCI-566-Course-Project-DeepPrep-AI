"""
data_loader.py
--------------
Loads the RecruitView dataset from HuggingFace and provides utilities
to iterate over video samples with their target labels.

Key improvements over v1:
  - ImageNet normalization applied to frames (required for pretrained EfficientNet)
  - Face-aware augmentation: brightness/contrast jitter, horizontal flip
    with landmark x-coordinate mirroring so geometry features stay valid
    - FaceFeatureExtractor is lazily instantiated per worker process
        (Windows-safe with DataLoader multiprocessing)
  - Augmentation only applied during training, not validation
"""

from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import cv2
import mediapipe as mp
from typing import Optional, List
import random

# ── ImageNet normalization stats (required for pretrained EfficientNet) ───────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Target metric column names ────────────────────────────────────────────────
TARGET_COLUMNS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
    "overall_personality",
    "communication",
    "confidence",
    "engagement",
    "clarity",
    "professional_appearance",
    "overall_performance",
]

# Eye contact + facial expression focused subset
VISUAL_TARGETS = [
    "extraversion",
    "confidence",
    "engagement",
    "professional_appearance",
    "overall_performance",
]


def build_frame_transform(img_size: int, augment: bool) -> T.Compose:
    """
    Build torchvision transform pipeline for a single frame.

    Training augmentation is designed to be face-safe:
      - ColorJitter: changes lighting/color, doesn't distort geometry
      - RandomHorizontalFlip: handled separately at the dataset level
        so we can also flip MediaPipe landmark x-coordinates
      - NO RandomCrop/Rotate: would move the face out of frame
    """
    transforms = [T.ToPILImage()]
    if augment:
        transforms += [
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.05),  # occasional grayscale for robustness
        ]
    transforms += [
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(transforms)


class RecruitViewDataset(Dataset):
    """
    PyTorch Dataset wrapping the RecruitView HuggingFace dataset.

    Each item returns:
        frames  : Tensor[T, 3, H, W]  – normalized frames (ImageNet stats)
        features: Tensor[T, F]        – MediaPipe facial geometry features
        labels  : Tensor[num_targets] – continuous target scores
    """

    def __init__(
        self,
        split: str = "train",
        num_frames: int = 24,
        img_size: int = 224,
        target_cols: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        augment: bool = True,
    ):
        print(f"Loading RecruitView ({split} split)...")
        self.dataset = load_dataset("AI4A-lab/RecruitView", split=split)
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

        self.num_frames = num_frames
        self.img_size = img_size
        self.augment = augment
        self.target_cols = target_cols or VISUAL_TARGETS
        self.transform = build_frame_transform(img_size, augment)

        # Lazily created in each worker process to avoid Windows spawn pickling errors.
        self.face_extractor = None

        # Validate target columns
        sample = self.dataset[0]
        missing = [c for c in self.target_cols if c not in sample]
        if missing:
            print(f"WARNING: columns not found: {missing}")
            print(f"Available keys: {list(sample.keys())}")
            self.target_cols = self._infer_target_cols(sample)
            print(f"Auto-detected targets: {self.target_cols}")

        print(f"Dataset ready: {len(self.dataset)} samples | "
              f"augment={augment} | targets={self.target_cols}")

    def _infer_target_cols(self, sample: dict) -> List[str]:
        skip = {"id", "user_no", "question_id", "video_quality", "duration"}
        return [
            k for k, v in sample.items()
            if k not in skip and isinstance(v, (int, float)) and not isinstance(v, bool)
        ]

    def __len__(self):
        return len(self.dataset)

    def __getstate__(self):
        # Drop MediaPipe runtime objects before worker-process pickling.
        state = self.__dict__.copy()
        state["face_extractor"] = None
        return state

    def _get_face_extractor(self):
        if self.face_extractor is None:
            self.face_extractor = FaceFeatureExtractor()
        return self.face_extractor

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Decide flip augmentation once per sample so frames + landmarks are consistent
        do_flip = self.augment and random.random() < 0.5

        frames_tensor, feat_tensor = self._process_video(sample["video"], do_flip)

        labels = []
        for col in self.target_cols:
            val = sample.get(col, 0.0)
            labels.append(float(val) if val is not None else 0.0)

        return {
            "frames": frames_tensor,                               # [T, 3, H, W]
            "features": feat_tensor,                               # [T, F]
            "labels": torch.tensor(labels, dtype=torch.float32),   # [num_targets]
            "id": sample.get("id", idx),
        }

    def _process_video(self, video_obj, do_flip: bool):
        total_frames = video_obj._num_frames
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        extractor = self._get_face_extractor()

        frames_list, features_list = [], []

        for i in indices:
            # frame_np is uint8 RGB, shape [H, W, 3]
            # torchcodec may return float32 in some versions — ensure uint8
            # so T.ToPILImage() always receives the format it expects
            raw = video_obj[int(i)].data.permute(1, 2, 0)
            if raw.dtype != torch.uint8:
                raw = (raw * 255).clamp(0, 255).to(torch.uint8)
            frame_np = raw.numpy()

            # MediaPipe runs on original (un-flipped) frame; flip flag mirrors x-coords
            feats = extractor.extract(frame_np, flip=do_flip)
            features_list.append(feats)

            # Visual augmentation + normalization
            if do_flip:
                frame_np = cv2.flip(frame_np, 1)  # horizontal flip

            frame_tensor = self.transform(frame_np)  # ColorJitter + Normalize
            frames_list.append(frame_tensor)

        return torch.stack(frames_list), torch.stack(features_list)


class FaceFeatureExtractor:
    """
    Extracts per-frame facial geometry features using MediaPipe FaceMesh.

    Feature vector per frame (total = FEATURE_DIM):
      - 478 landmark (x, y, z) coordinates (refine_landmarks adds 10 iris pts)
      - Eye aspect ratios left + right  (blink / alertness)
      - Mouth aspect ratio              (expression proxy)
      - Gaze offset x, y               (eye-center vs nose-tip → eye contact proxy)
      - Head yaw proxy                  (how much face turns away from camera)

    All values are normalized to [0, 1] or [-1, 1] by MediaPipe's coordinate system.
    Zero-vector returned when no face is detected.
    """

    NUM_LANDMARKS = 478  # 468 + 10 iris refinement points
    EXTRA_FEATURES = 6
    FEATURE_DIM = NUM_LANDMARKS * 3 + EXTRA_FEATURES  # 1434 + 6 = 1440

    # Key landmark indices (MediaPipe FaceMesh with refine_landmarks=True)
    LEFT_EYE_TOP, LEFT_EYE_BOT   = 159, 145
    RIGHT_EYE_TOP, RIGHT_EYE_BOT = 386, 374
    LEFT_EYE_INNER, LEFT_EYE_OUTER   = 133,  33
    RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263
    NOSE_TIP    = 4
    UPPER_LIP   = 13
    LOWER_LIP   = 14
    LEFT_MOUTH  = 61
    RIGHT_MOUTH = 291
    LEFT_CHEEK  = 234
    RIGHT_CHEEK = 454

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )

    def extract(self, frame_rgb: np.ndarray, flip: bool = False) -> torch.Tensor:
        """
        Args:
            frame_rgb : H×W×3 uint8 RGB image (original, NOT yet flipped)
            flip      : If True, mirror all x-coordinates so landmarks match
                        the horizontally flipped frame

        Returns:
            FloatTensor [FEATURE_DIM]
        """
        result = self.face_mesh.process(frame_rgb)

        if not result.multi_face_landmarks:
            return torch.zeros(self.FEATURE_DIM, dtype=torch.float32)

        lms = result.multi_face_landmarks[0].landmark
        coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)  # [478, 3]

        if flip:
            coords[:, 0] = 1.0 - coords[:, 0]

        flat = coords.flatten()  # [1434]

        # ── Derived geometric features ────────────────────────────────────────
        def pt(idx):
            return coords[idx]

        # Eye Aspect Ratio
        left_ear  = _dist(pt(self.LEFT_EYE_TOP),  pt(self.LEFT_EYE_BOT))  \
                  / (_dist(pt(self.LEFT_EYE_INNER), pt(self.LEFT_EYE_OUTER)) + 1e-6)
        right_ear = _dist(pt(self.RIGHT_EYE_TOP), pt(self.RIGHT_EYE_BOT)) \
                  / (_dist(pt(self.RIGHT_EYE_INNER), pt(self.RIGHT_EYE_OUTER)) + 1e-6)

        # Mouth Aspect Ratio
        mouth_ratio = _dist(pt(self.UPPER_LIP), pt(self.LOWER_LIP)) \
                    / (_dist(pt(self.LEFT_MOUTH), pt(self.RIGHT_MOUTH)) + 1e-6)

        # Gaze offset
        left_eye_cx  = (pt(self.LEFT_EYE_INNER)[0]  + pt(self.LEFT_EYE_OUTER)[0])  / 2
        right_eye_cx = (pt(self.RIGHT_EYE_INNER)[0] + pt(self.RIGHT_EYE_OUTER)[0]) / 2
        left_eye_cy  = (pt(self.LEFT_EYE_INNER)[1]  + pt(self.LEFT_EYE_OUTER)[1])  / 2
        right_eye_cy = (pt(self.RIGHT_EYE_INNER)[1] + pt(self.RIGHT_EYE_OUTER)[1]) / 2
        gaze_x = ((left_eye_cx + right_eye_cx) / 2) - pt(self.NOSE_TIP)[0]
        gaze_y = ((left_eye_cy + right_eye_cy) / 2) - pt(self.NOSE_TIP)[1]

        # Head yaw proxy
        left_to_nose  = pt(self.NOSE_TIP)[0] - pt(self.LEFT_CHEEK)[0]
        right_to_nose = pt(self.RIGHT_CHEEK)[0] - pt(self.NOSE_TIP)[0]
        yaw_proxy = (left_to_nose - right_to_nose) / (left_to_nose + right_to_nose + 1e-6)

        derived = np.array(
            [left_ear, right_ear, mouth_ratio, gaze_x, gaze_y, yaw_proxy],
            dtype=np.float32
        )

        return torch.from_numpy(np.concatenate([flat, derived]))


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def get_dataloaders(cfg):
    """
    Build train/val DataLoaders from a TrainConfig object.
    Does an 80/20 split since RecruitView ships as a single 'train' split.

    Note on num_workers: On Windows, multiprocessing DataLoader workers use
    'spawn', which requires every object in the Dataset to be picklable.
    Both MediaPipe (C++ runtime) and HuggingFace datasets (internal file handles)
    fail this check. num_workers is forced to 0 on Windows so everything runs
    in the main process. On Linux/Mac, lazy MediaPipe init + __getstate__ is
    sufficient and num_workers > 0 works fine.
    """
    import platform
    from torch.utils.data import DataLoader, random_split

    safe_workers = 0 if platform.system() == "Windows" else cfg.num_workers
    if safe_workers != cfg.num_workers:
        print(f"  Windows detected: setting num_workers=0 (was {cfg.num_workers})")

    full_dataset = RecruitViewDataset(
        split="train",
        num_frames=cfg.num_frames,
        img_size=cfg.img_size,
        target_cols=cfg.target_cols,
        max_samples=cfg.debug_samples if cfg.debug else None,
        augment=True,
    )

    n_total = len(full_dataset)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Separate val dataset with augment=False for clean evaluation
    val_dataset = RecruitViewDataset(
        split="train",
        num_frames=cfg.num_frames,
        img_size=cfg.img_size,
        target_cols=full_dataset.target_cols,
        max_samples=cfg.debug_samples if cfg.debug else None,
        augment=False,
    )
    val_ds_clean = torch.utils.data.Subset(val_dataset, val_ds.indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=safe_workers,
        pin_memory=cfg.pin_memory and safe_workers > 0,
        persistent_workers=safe_workers > 0,
        prefetch_factor=2 if safe_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds_clean,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=safe_workers,
        pin_memory=cfg.pin_memory and safe_workers > 0,
        persistent_workers=safe_workers > 0,
        prefetch_factor=2 if safe_workers > 0 else None,
    )

    print(f"Train: {n_train} samples | Val: {n_val} samples")
    return train_loader, val_loader, full_dataset.target_cols
