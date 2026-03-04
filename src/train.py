import torch
import time
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

from analysis import calculate_ece

def train_and_evaluate(model_name, train_ds, val_ds, test_ds, test_df, name_label, candidate_labels):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
    )

    def tokenize_func(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_train = train_ds.map(tokenize_func, batched=True)
    tokenized_val = val_ds.map(tokenize_func, batched=True)
    tokenized_test = test_ds.map(tokenize_func, batched=True)

    training_args = TrainingArguments(
        output_dir=f"./results_{name_label}",
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        report_to="none"
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    # Evaluation phase
    print(f"Evaluating {name_label} on Test Set...")
    start_time = time.time()
    raw_predictions = trainer.predict(tokenized_test)
    end_time = time.time()

    logits = raw_predictions.predictions
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    y_true = test_df["label"].values
    time_per_100 = ((end_time - start_time) / len(test_df)) * 100
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average='macro')
    ece = calculate_ece(y_true, preds, confidences)

    class_report_str = classification_report(y_true, preds, target_names=candidate_labels)
    class_report_dict = classification_report(y_true, preds, target_names=candidate_labels, output_dict=True)

    return {"Accuracy": acc, "Macro F1": f1, "ECE": ece, "Time/100": time_per_100, "Classification Report": class_report_str, "Classification Report Dict": class_report_dict}
