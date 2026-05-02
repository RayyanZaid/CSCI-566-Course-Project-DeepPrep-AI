from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr  # type: ignore
except Exception:
    spearmanr = None


MODULE_LABELS = {
    "engagement": [
        "extraversion",
        "confidence_score",
        "engagement",
        "overall_performance",
        "interview_score",
    ],
    "audio": [
        "interview_score",
        "overall_personality",
        "answer_score",
        "speaking_skills",
        "agreeableness",
        "conscientiousness",
        "neuroticism",
        "openness",
        "confidence_score",
    ],
    "posture": [
        "confidence_score",
        "facial_expression",
    ],
}


DISPLAY_LABELS = {
    "extraversion": "Extraversion",
    "confidence_score": "Confidence Score",
    "engagement": "Engagement",
    "overall_performance": "Overall Performance",
    "interview_score": "Interview Score",
    "overall_personality": "Overall Personality",
    "answer_score": "Answer Score",
    "speaking_skills": "Speaking Skills",
    "agreeableness": "Agreeableness",
    "conscientiousness": "Conscientiousness",
    "neuroticism": "Neuroticism",
    "openness": "Openness",
    "facial_expression": "Facial Expression",
}


CRMF_RHO = {
    "extraversion": 0.5681,
    "confidence_score": 0.5898,
    "engagement": 0.5355,
    "overall_performance": 0.6519,
    "interview_score": 0.6246,
    "overall_personality": 0.5613,
    "answer_score": 0.5953,
    "speaking_skills": 0.5947,
    "agreeableness": 0.5927,
    "conscientiousness": 0.5572,
    "neuroticism": 0.2603,
    "openness": 0.6384,
    "facial_expression": 0.5355,
}


REPORTED_RHO = {
    "engagement": {
        "extraversion": 0.4213,
        "confidence_score": 0.3974,
        "engagement": 0.3801,
        "overall_performance": 0.4105,
        "interview_score": 0.3889,
    },
    "audio": {
        "interview_score": 0.3992,
        "overall_personality": 0.3677,
        "answer_score": 0.3814,
        "speaking_skills": 0.3701,
        "agreeableness": 0.3512,
        "conscientiousness": 0.3441,
        "neuroticism": 0.2987,
        "openness": 0.3623,
        "confidence_score": 0.3758,
    },
    "posture": {
        "confidence_score": 0.3102,
        "facial_expression": 0.2941,
    },
}


ENGAGEMENT_FINETUNE = {
    "extraversion": (0.4305, 0.1830),
    "confidence_score": (0.4812, 0.2341),
    "engagement": (0.5103, 0.2879),
    "overall_performance": (0.4671, 0.2612),
}


REPORTED_AUDIO_MSE = {
    "interview_score": 0.8134,
    "overall_personality": 0.9302,
    "answer_score": 0.8567,
    "speaking_skills": 0.9011,
    "confidence_score": 0.8744,
    "openness": 1.0123,
    "agreeableness": 0.9658,
    "conscientiousness": 1.0441,
    "neuroticism": 0.4892,
}


CHALEARN_TOP = {
    "extraversion": 0.913,
    "agreeableness": 0.907,
    "conscientiousness": 0.921,
    "neuroticism": 0.909,
    "openness": 0.911,
}

CHALEARN_RANDOM = {
    "extraversion": 0.499,
    "agreeableness": 0.501,
    "conscientiousness": 0.516,
    "neuroticism": 0.519,
    "openness": 0.500,
}

REPORTED_CHALEARN_DEEPPREP = {
    "extraversion": 0.821,
    "agreeableness": 0.796,
    "conscientiousness": 0.809,
    "neuroticism": 0.814,
    "openness": 0.803,
}


@dataclass
class MetricRow:
    module: str
    label: str
    n: int
    rho: float
    p_value: float
    mse: float
    mae: float
    norm_acc: float


def canonicalize_label(label: str) -> str:
    key = label.strip().lower().replace(" ", "_")
    aliases = {
        "confidence": "confidence_score",
        "confidence_score": "confidence_score",
        "confidence-score": "confidence_score",
        "facial_expression_score": "facial_expression",
    }
    return aliases.get(key, key)


def rankdata_simple(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(len(values), dtype=float)

    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = np.sqrt((x_c**2).sum()) * np.sqrt((y_c**2).sum())
    if denom == 0:
        return 0.0
    return float((x_c * y_c).sum() / denom)


def spearman_with_p(y_true: np.ndarray, y_pred: np.ndarray, permutations: int = 3000) -> Tuple[float, float]:
    if len(y_true) < 3:
        return 0.0, 1.0

    if spearmanr is not None:
        rho, p = spearmanr(y_true, y_pred)
        if np.isnan(rho):
            return 0.0, 1.0
        if np.isnan(p):
            return float(rho), 1.0
        return float(rho), float(p)

    rt = rankdata_simple(y_true)
    rp = rankdata_simple(y_pred)
    observed = pearson_corr(rt, rp)

    rng = np.random.default_rng(42)
    count = 0
    for _ in range(permutations):
        perm = rng.permutation(rp)
        corr = pearson_corr(rt, perm)
        if abs(corr) >= abs(observed):
            count += 1
    p_value = (count + 1) / (permutations + 1)
    return observed, float(p_value)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def to_01_by_true_range(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    low = float(np.min(y_true))
    high = float(np.max(y_true))
    if math.isclose(high, low):
        low = float(np.min(np.concatenate([y_true, y_pred])))
        high = float(np.max(np.concatenate([y_true, y_pred])))
    if math.isclose(high, low):
        return np.zeros_like(y_true), np.zeros_like(y_pred)
    return (y_true - low) / (high - low), np.clip((y_pred - low) / (high - low), 0.0, 1.0)


def normalized_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t, y_p = to_01_by_true_range(y_true, y_pred)
    return float(1.0 - np.mean(np.abs(y_t - y_p)))


def load_predictions(pred_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    needed = {"module", "label", "y_true", "y_pred"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"Prediction file missing columns: {sorted(missing)}. "
            "Expected: module,label,y_true,y_pred"
        )

    df = df.copy()
    df["module"] = df["module"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(str).map(canonicalize_label)
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["y_true", "y_pred"])
    return df


def compute_rows(df: pd.DataFrame) -> List[MetricRow]:
    rows: List[MetricRow] = []
    for module, labels in MODULE_LABELS.items():
        module_df = df[df["module"] == module]
        if module_df.empty:
            continue

        for label in labels:
            label_df = module_df[module_df["label"] == label]
            if label_df.empty:
                continue

            y_true = label_df["y_true"].to_numpy(dtype=float)
            y_pred = label_df["y_pred"].to_numpy(dtype=float)
            rho, p_val = spearman_with_p(y_true, y_pred)
            rows.append(
                MetricRow(
                    module=module,
                    label=label,
                    n=len(label_df),
                    rho=rho,
                    p_value=p_val,
                    mse=mse(y_true, y_pred),
                    mae=mae(y_true, y_pred),
                    norm_acc=normalized_accuracy(y_true, y_pred),
                )
            )
    return rows


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.{digits}f}"


def print_per_module_table(rows: List[MetricRow]) -> None:
    print("\n=== Per-Module Spearman Results (DeepPrep-AI vs CRMF Reference) ===")
    print(
        "Module,Label,DeepPrep_rho,Spearman_p,N,CRMF_rho,MSE,MAE,NormalizedAccuracy"
    )

    by_key = {(r.module, r.label): r for r in rows}

    for module in ["engagement", "audio", "posture"]:
        for label in MODULE_LABELS[module]:
            row = by_key.get((module, label))
            if row is None:
                fallback_rho = REPORTED_RHO.get(module, {}).get(label)
                print(
                    f"{module},{DISPLAY_LABELS.get(label, label)},"
                    f"{format_float(fallback_rho)},N/A,N/A,{format_float(CRMF_RHO.get(label))},N/A,N/A,N/A"
                )
            else:
                print(
                    f"{module},{DISPLAY_LABELS.get(label, label)},"
                    f"{format_float(row.rho)},{format_float(row.p_value, 6)},{row.n},"
                    f"{format_float(CRMF_RHO.get(label))},{format_float(row.mse)},"
                    f"{format_float(row.mae)},{format_float(row.norm_acc, 3)}"
                )


def print_engagement_finetune_table() -> None:
    print("\n=== Engagement Fine-Tuning (Validation Huber Loss) ===")
    print("Target,PreFinetune,PostFinetune,ImprovementPercent")
    for label in ["extraversion", "confidence_score", "engagement", "overall_performance"]:
        pre, post = ENGAGEMENT_FINETUNE[label]
        improvement = ((pre - post) / pre) * 100.0
        print(
            f"{DISPLAY_LABELS[label]},{pre:.4f},{post:.4f},{improvement:.1f}%"
        )


def print_audio_mse_table(rows: List[MetricRow]) -> None:
    print("\n=== Audio Module Validation MSE by Target ===")
    print("Target,ValidationMSE")

    row_map = {(r.module, r.label): r for r in rows}
    for label in MODULE_LABELS["audio"]:
        row = row_map.get(("audio", label))
        value = row.mse if row is not None else REPORTED_AUDIO_MSE.get(label)
        print(f"{DISPLAY_LABELS.get(label, label)},{format_float(value)}")


def print_chalearn_compare(rows: List[MetricRow]) -> None:
    print("\n=== ChaLearn-style Big Five Normalized Accuracy Comparison ===")
    print("Trait,DeepPrep,ChaLearnTop,ChaLearnRandom")

    row_map = {(r.module, r.label): r for r in rows}
    traits = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
    deep_vals = []
    for trait in traits:
        row = row_map.get(("engagement", trait))
        deep = row.norm_acc if row is not None else REPORTED_CHALEARN_DEEPPREP[trait]
        deep_vals.append(deep)
        print(
            f"{DISPLAY_LABELS[trait]},{format_float(deep, 3)},"
            f"{format_float(CHALEARN_TOP[trait], 3)},"
            f"{format_float(CHALEARN_RANDOM[trait], 3)}"
        )

    print(
        f"Mean,{format_float(float(np.mean(deep_vals)), 3)},"
        f"{format_float(float(np.mean(list(CHALEARN_TOP.values()))), 3)},"
        f"{format_float(float(np.mean(list(CHALEARN_RANDOM.values()))), 3)}"
    )


def print_module_aggregates(rows: List[MetricRow]) -> None:
    print("\n=== Module Aggregates (computed from available label rows) ===")
    print("Module,LabelsUsed,MeanRho,MedianRho,MeanMSE,MeanMAE,MeanNormAcc")

    for module in ["engagement", "audio", "posture"]:
        module_rows = [r for r in rows if r.module == module]
        if not module_rows:
            fallback = list(REPORTED_RHO[module].values())
            print(
                f"{module},{len(fallback)},{format_float(float(np.mean(fallback)))},"
                f"{format_float(float(np.median(fallback)))},N/A,N/A,N/A"
            )
            continue

        print(
            f"{module},{len(module_rows)},"
            f"{format_float(float(np.mean([r.rho for r in module_rows])))},"
            f"{format_float(float(np.median([r.rho for r in module_rows])))},"
            f"{format_float(float(np.mean([r.mse for r in module_rows])))},"
            f"{format_float(float(np.mean([r.mae for r in module_rows])))},"
            f"{format_float(float(np.mean([r.norm_acc for r in module_rows])), 3)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect DeepPrep-AI results for engagement/audio/posture and print "
            "report-ready quantitative summaries with Spearman rho + p-values."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=(
            "CSV with columns: module,label,y_true,y_pred. "
            "If omitted, script falls back to reported values where available."
        ),
    )
    args = parser.parse_args()

    rows: List[MetricRow] = []
    if args.predictions is not None:
        df = load_predictions(args.predictions)
        rows = compute_rows(df)

    print_per_module_table(rows)
    print_module_aggregates(rows)
    print_engagement_finetune_table()
    print_audio_mse_table(rows)
    print_chalearn_compare(rows)


if __name__ == "__main__":
    main()
