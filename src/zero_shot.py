from transformers import pipeline
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from analysis import calculate_ece

def run_zero_shot(df, classifier, candidate_labels, hypothesis_template):

    texts = df["text"].tolist()

    predictions = classifier(
        texts,
        candidate_labels,
        hypothesis_template=hypothesis_template,
        batch_size=16
    )

    # Extract the top predicted label
    return [pred['labels'][0] for pred in predictions]


DistilBERT_zero_shot_pipeline = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
    device=0
)

DistilRoBERTa_zero_shot_pipeline = pipeline(
    "zero-shot-classification",
    model="cross-encoder/nli-distilroberta-base",
    device=0
)


def run_benchmarked_inference(df, classifier, candidate_labels, label_id, hypothesis_template):
    texts = df["text"].tolist()

    start_time = time.time()
    
    # We need the full results to get probabilities for ECE calculation
    raw_results = classifier(
        texts,
        candidate_labels,
        hypothesis_template=hypothesis_template,
        batch_size=16
    )

    end_time = time.time()

    # calculate time per 100 samples.
    total_time = end_time - start_time
    time_per_100 = (total_time / len(texts)) * 100

    # Extract predicted labels and their confidences.
    preds_labels = [res['labels'][0] for res in raw_results]
    preds_ids = [label_id[label] for label in preds_labels]
    confidences = [res['scores'][0] for res in raw_results]

    # calculate metrics
    y_true = df["label"].values
    acc = accuracy_score(y_true, preds_ids)
    f1 = f1_score(y_true, preds_ids, average='macro')
    ece = calculate_ece(y_true, np.array(preds_ids), np.array(confidences))

    class_report = classification_report(y_true, preds_ids, target_names=candidate_labels)

    return {
        "Accuracy": acc,
        "Macro F1": f1,
        "ECE": ece,
        "Time/100": time_per_100,
        "Classification Report": class_report
    }
