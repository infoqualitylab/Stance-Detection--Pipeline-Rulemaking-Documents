"""
Stance Detection using NLI RoBERTa
------------------------------------
Uses the label_stance column as the per-row topic target.
Compares predictions against stance_pred (ground truth).

Columns required in CSV:
  - text          : the input text
  - label_stance  : the claim/topic to detect stance toward (per row)
  - stance_pred   : ground-truth labels (Oppose / Neutral / Support)

Usage:
    pip install transformers torch pandas scikit-learn tqdm

    python3 RoBERTA.py --input model_ready.csv
    python3 RoBERTA.py --input model_ready.csv --output results.csv
"""

import os
import argparse
import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import warnings

# Suppress TensorFlow AVX crash
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

MODEL_NAME = "cross-encoder/nli-roberta-base"

LABEL_NORMALISE = {
    "support": "Support",
    "oppose":  "Oppose",
    "neutral": "Neutral",
    "against": "Oppose",
    "favor":   "Support",
    "favour":  "Support",
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["text", "label_stance", "stance_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

    df["stance_pred"] = df["stance_pred"].str.strip().str.lower().map(LABEL_NORMALISE)
    invalid = df["stance_pred"].isna().sum()
    if invalid:
        print(f"[WARNING] Dropping {invalid} rows with unrecognised stance labels.")
        df = df.dropna(subset=["stance_pred"])

    return df.reset_index(drop=True)


def predict_stance(classifier, df: pd.DataFrame) -> list[str]:
    """Run zero-shot NLI stance detection using label_stance as the topic per row."""
    predictions = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        topic = row["label_stance"]
        candidate_labels = [
            f"The author opposes or raises concerns about: {topic}",
            f"The author supports or agrees with: {topic}",
            f"The author is neutral or descriptive about: {topic}",
        ]
        label_map = {
            candidate_labels[0]: "Oppose",
            candidate_labels[1]: "Support",
            candidate_labels[2]: "Neutral",
        }
        result = classifier(
            sequences=row["text"],
            candidate_labels=candidate_labels,
            hypothesis_template="{}.",
            multi_label=False,
        )
        predictions.append(label_map[result["labels"][0]])

    return predictions


def evaluate(y_true: list, y_pred: list) -> None:
    labels = ["Support", "Oppose", "Neutral"]
    acc = accuracy_score(y_true, y_pred)
    correct = sum(t == p for t, p in zip(y_true, y_pred))

    print("\n" + "=" * 45)
    print(f"  Overall Accuracy: {acc * 100:.2f}%  ({correct}/{len(y_true)} correct)")
    print("=" * 45)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    print("Confusion Matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())


def accuracy_by_topic(df: pd.DataFrame) -> None:
    print("\n\nAccuracy by label_stance:")
    print("-" * 70)
    for topic, group in df.groupby("label_stance"):
        acc = accuracy_score(group["stance_pred"], group["roberta_stance"])
        print(f"  {topic[:65]:<65}  {acc * 100:.1f}%  (n={len(group)})")


def main():
    parser = argparse.ArgumentParser(description="NLI RoBERTa Stance Detection")
    parser.add_argument("--input",  "-i", required=True, help="Path to input CSV")
    parser.add_argument("--output", "-o", default="stance_results.csv")
    parser.add_argument("--model",  default=MODEL_NAME)
    parser.add_argument("--device", type=int, default=-1, help="-1=CPU, 0=GPU")
    args = parser.parse_args()

    # Load
    print(f"Loading: {args.input}")
    df = load_data(args.input)
    print(f"  {len(df)} rows loaded")
    print(f"\n  Ground-truth distribution:\n{df['stance_pred'].value_counts().to_string()}")

    # Device
    device = args.device
    if device == -1 and torch.cuda.is_available():
        device = 0
        print("\nGPU detected — using CUDA.")
    else:
        print("\nRunning on CPU.")

    # Load model
    print(f"\nLoading model: {args.model}")
    classifier = pipeline(
        "zero-shot-classification",
        model=args.model,
        device=device,
    )

    # Predict
    predictions = predict_stance(classifier, df)

    # Attach predictions
    df["roberta_stance"] = predictions
    df["correct"] = df["stance_pred"] == df["roberta_stance"]

    # Evaluate
    evaluate(df["stance_pred"].tolist(), predictions)
    accuracy_by_topic(df)

    # Save
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()