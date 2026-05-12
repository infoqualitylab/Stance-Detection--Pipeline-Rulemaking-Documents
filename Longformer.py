"""
Stance Detection using Longformer

Longformer supports up to 4096 tokens, making it suitable for the
long policy texts in your dataset — no chunking required.

Columns required in CSV:
  - text          : the input text
  - label_stance  : the claim/topic to detect stance toward (per row)
  - stance_pred   : ground-truth labels (Oppose / Neutral / Support)

Usage:
    pip install transformers torch pandas scikit-learn tqdm

    python3 stance_longformer.py --input model_ready.csv
    python3 stance_longformer.py --input model_ready.csv --output results.csv
"""

import os
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from transformers import LongformerConfig
import warnings

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from transformers import LongformerConfig



MODEL_NAME = "allenai/longformer-base-4096"

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


def nli_score(model, tokenizer, text: str, hypothesis: str, device: torch.device) -> dict:
    """
    Encode a (text, hypothesis) pair and return entailment probabilities.
    Longformer requires global_attention_mask with attention on [CLS] token.
    MNLI label order: 0=contradiction, 1=neutral, 2=entailment.
    """
    encoding = tokenizer(
        text,
        hypothesis,
        return_tensors="pt",
        max_length=4096,
        truncation="only_first",   # truncate text, never the hypothesis
        padding="max_length",
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Global attention on [CLS] token — required by Longformer
    global_attention_mask = torch.zeros_like(input_ids)
    global_attention_mask[:, 0] = 1

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )

    probs = F.softmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
    return {
        "contradiction": probs[0],
        "neutral":       probs[1],
        "entailment":    probs[2],
    }


def predict_stance(model, tokenizer, df: pd.DataFrame, device: torch.device) -> list[str]:
    """
    For each row, score three hypotheses (Oppose / Support / Neutral)
    and assign the stance with the highest entailment probability.
    """
    predictions = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        topic = row["label_stance"]
        hypotheses = {
            "Oppose":  f"The author opposes or raises concerns about: {topic}",
            "Support": f"The author supports or agrees with: {topic}",
            "Neutral": f"The author is neutral or descriptive about: {topic}",
        }

        scores = {
            stance: nli_score(model, tokenizer, str(row["text"]), hyp, device)["entailment"]
            for stance, hyp in hypotheses.items()
        }
        predictions.append(max(scores, key=scores.get))

    return predictions


def evaluate(y_true: list, y_pred: list) -> None:
    labels = ["Support", "Oppose", "Neutral"]
    acc = accuracy_score(y_true, y_pred)
    correct = sum(t == p for t, p in zip(y_true, y_pred))

    print("\n" + "=" * 55)
    print(f"  Overall Accuracy: {acc * 100:.2f}%  ({correct}/{len(y_true)} correct)")
    print("=" * 55)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    print("Confusion Matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())


def accuracy_by_topic(df: pd.DataFrame) -> None:
    print("\n\nAccuracy by label_stance:")
    print("-" * 70)
    for topic, group in df.groupby("label_stance"):
        acc = accuracy_score(group["stance_pred"], group["longformer_stance"])
        print(f"  {topic[:65]:<65}  {acc * 100:.1f}%  (n={len(group)})")


def main():
    parser = argparse.ArgumentParser(description="Longformer NLI Stance Detection")
    parser.add_argument("--input",  "-i", required=True)
    parser.add_argument("--output", "-o", default="stance_results_longformer.csv")
    parser.add_argument("--model",  default=MODEL_NAME)
    parser.add_argument("--device", type=int, default=-1, help="-1=CPU, 0=GPU")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    df = load_data(args.input)
    print(f"  {len(df)} rows loaded")
    print(f"\n  Ground-truth distribution:\n{df['stance_pred'].value_counts().to_string()}")

    device = torch.device("cuda" if (args.device >= 0 and torch.cuda.is_available()) else "cpu")
    print(f"\nUsing device: {device}")

    print(f"\nLoading Longformer: {args.model}")
    print("  Max tokens: 4096 — long texts handled natively, no chunking needed")
    tokenizer = LongformerTokenizer.from_pretrained(args.model)
    model = LongformerForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
         use_safetensors=True
    ).to(device)
    model.eval()

    predictions = predict_stance(model, tokenizer, df, device)

    df["longformer_stance"] = predictions
    df["correct"] = df["stance_pred"] == df["longformer_stance"]

    evaluate(df["stance_pred"].tolist(), predictions)
    accuracy_by_topic(df)

    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()