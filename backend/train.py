"""
Training script for sentiment analysis models.
Can be run locally or on AWS EC2/SageMaker.

Usage:
    python train.py --model hmgs --epochs 5 --batch-size 16
    python train.py --model bilstm --epochs 5 --batch-size 16
"""

import argparse
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, classification_report
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from app.config import config
from app.models.sentiment_models import HMGS, BertBiLSTMAttention, BertLinearBaseline
from app.services.aws_service import upload_model_to_s3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReviewDataset(Dataset):
    """Segments reviews into sentences for HMGS model."""

    def __init__(self, texts, labels, tokenizer, max_len=128, max_sents=10):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len
        self.max_sents = max_sents

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        sents = sent_tokenize(text)[:self.max_sents]
        n = len(sents)

        ids_all, mask_all = [], []
        for s in sents:
            enc = self.tok(s, max_length=self.max_len, padding="max_length",
                           truncation=True, return_tensors="pt")
            ids_all.append(enc["input_ids"].squeeze(0))
            mask_all.append(enc["attention_mask"].squeeze(0))

        pad = self.max_sents - n
        if pad > 0:
            ids_all.append(torch.zeros(pad, self.max_len, dtype=torch.long))
            mask_all.append(torch.zeros(pad, self.max_len, dtype=torch.long))

        input_ids = torch.cat([t.unsqueeze(0) if t.dim() == 1 else t for t in ids_all], dim=0)
        attn_mask = torch.cat([t.unsqueeze(0) if t.dim() == 1 else t for t in mask_all], dim=0)
        sent_mask = torch.zeros(self.max_sents, dtype=torch.long)
        sent_mask[:n] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "sent_mask": sent_mask,
            "doc_label": torch.tensor(label, dtype=torch.long),
            "sent_labels": torch.full((self.max_sents,), label, dtype=torch.long),
        }


class FlatReviewDataset(Dataset):
    """Flat tokenization for BiLSTM/Linear models."""

    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            str(self.texts[idx]), max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def generate_training_data(n_samples=1000, seed=42):
    """Generate synthetic training data for demonstration."""
    set_seed(seed)
    templates = {
        0: [  # Negative
            "Terrible product. {} stopped working after {} days.",
            "Very disappointed with the {}. {} was awful.",
            "Do not buy this. The {} is terrible and {} is worse.",
            "Worst {} I have ever used. {} is completely broken.",
            "The {} quality is unacceptable. {} keeps failing.",
        ],
        1: [  # Neutral
            "The {} is okay. {} could be better but it's functional.",
            "Average {}. {} meets basic expectations but nothing more.",
            "Decent {} for the price. {} is mediocre.",
            "The {} works fine. {} is neither great nor terrible.",
            "It's an acceptable {}. {} does its job.",
        ],
        2: [  # Positive
            "Love the {}! {} is absolutely fantastic.",
            "Best {} I have ever bought. {} exceeds expectations.",
            "The {} quality is outstanding. {} is incredible.",
            "Amazing {}. {} works perfectly every single time.",
            "Excellent {} with great {}. Highly recommend!",
        ],
    }
    aspects = ["battery life", "screen", "keyboard", "camera", "processor",
               "build quality", "sound", "design", "software", "price"]
    durations = ["3", "5", "7", "10", "14", "30"]

    texts, labels = [], []
    for i in range(n_samples):
        label = i % 3
        template = random.choice(templates[label])
        a1, a2 = random.sample(aspects, 2)
        if label == 0:
            text = template.format(a1, random.choice(durations))
        else:
            text = template.format(a1, a2)
        texts.append(text)
        labels.append(label)

    return texts, labels


def train_hmgs(args):
    """Train the HMGS model."""
    print("Training HMGS model...")
    set_seed(42)
    tokenizer = BertTokenizerFast.from_pretrained(config.model.bert_model_name)

    texts, labels = generate_training_data(args.train_samples)
    split = int(len(texts) * 0.8)
    train_ds = ReviewDataset(texts[:split], labels[:split], tokenizer)
    val_ds = ReviewDataset(texts[split:], labels[split:], tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    bert = BertModel.from_pretrained(config.model.bert_model_name)
    model = HMGS(bert, config.model).to(DEVICE)

    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": 2e-5},
        {"params": [p for n, p in model.named_parameters() if "bert" not in n], "lr": 1e-3},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    best_f1 = 0
    os.makedirs(config.model_path, exist_ok=True)
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            smask = batch["sent_mask"].to(DEVICE)
            doc_lab = batch["doc_label"].to(DEVICE)

            B, S, L = ids.shape
            flat_ids = ids.view(B * S, L)
            flat_mask = mask.view(B * S, L)
            real = smask.view(B * S).bool()
            h = config.model.hidden_dim
            all_cls = torch.zeros(B * S, h, device=DEVICE)
            if real.any():
                out = model.bert(input_ids=flat_ids[real], attention_mask=flat_mask[real])
                all_cls[real] = out.last_hidden_state[:, 0, :]

            doc_result = model.forward_doc(all_cls.view(B, S, -1), smask, doc_lab)
            loss = doc_result["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        preds, true_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                smask = batch["sent_mask"].to(DEVICE)
                doc_lab = batch["doc_label"].to(DEVICE)

                B, S, L = ids.shape
                flat_ids = ids.view(B * S, L)
                flat_mask = mask.view(B * S, L)
                real = smask.view(B * S).bool()
                all_cls = torch.zeros(B * S, h, device=DEVICE)
                if real.any():
                    out = model.bert(input_ids=flat_ids[real], attention_mask=flat_mask[real])
                    all_cls[real] = out.last_hidden_state[:, 0, :]

                doc_result = model.forward_doc(all_cls.view(B, S, -1), smask, doc_lab)
                preds.extend(doc_result["logits"].argmax(-1).cpu().numpy())
                true_labels.extend(doc_lab.cpu().numpy())
                val_loss += doc_result["loss"].item()

        val_f1 = f1_score(true_labels, preds, average="macro")
        val_acc = accuracy_score(true_labels, preds)
        avg_val_loss = val_loss / len(val_loader)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
            "val_f1": val_f1,
            "val_acc": val_acc,
        })

        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(config.model_path, "hmgs_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (F1={val_f1:.4f})")

    # Save training history
    with open(os.path.join(config.model_path, "hmgs_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Upload to S3 if configured
    if config.aws.access_key_id:
        upload_model_to_s3(
            os.path.join(config.model_path, "hmgs_best.pt"),
            "models/hmgs_best.pt"
        )

    print(f"\nTraining complete. Best F1: {best_f1:.4f}")
    return history


def train_bilstm(args):
    """Train the BERT-BiLSTM-Attention model."""
    print("Training BERT-BiLSTM-Attention model...")
    set_seed(42)
    tokenizer = BertTokenizerFast.from_pretrained(config.model.bert_model_name)

    texts, labels = generate_training_data(args.train_samples)
    split = int(len(texts) * 0.8)
    train_ds = FlatReviewDataset(texts[:split], labels[:split], tokenizer)
    val_ds = FlatReviewDataset(texts[split:], labels[split:], tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    bert = BertModel.from_pretrained(config.model.bert_model_name)
    model = BertBiLSTMAttention(bert).to(DEVICE)

    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": 2e-5},
        {"params": [p for n, p in model.named_parameters() if "bert" not in n], "lr": 1e-3},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    best_f1 = 0
    os.makedirs(config.model_path, exist_ok=True)
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lab = batch["labels"].to(DEVICE)

            result = model(ids, mask, lab)
            loss = result["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        preds, true_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                lab = batch["labels"].to(DEVICE)
                result = model(ids, mask, lab)
                preds.extend(result["logits"].argmax(-1).cpu().numpy())
                true_labels.extend(lab.cpu().numpy())
                val_loss += result["loss"].item()

        val_f1 = f1_score(true_labels, preds, average="macro")
        val_acc = accuracy_score(true_labels, preds)
        avg_val_loss = val_loss / len(val_loader)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
            "val_f1": val_f1,
            "val_acc": val_acc,
        })

        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(config.model_path, "bilstm_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (F1={val_f1:.4f})")

    with open(os.path.join(config.model_path, "bilstm_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if config.aws.access_key_id:
        upload_model_to_s3(
            os.path.join(config.model_path, "bilstm_best.pt"),
            "models/bilstm_best.pt"
        )

    print(f"\nTraining complete. Best F1: {best_f1:.4f}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train sentiment models")
    parser.add_argument("--model", choices=["hmgs", "bilstm", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=600)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.model in ("hmgs", "all"):
        train_hmgs(args)
    if args.model in ("bilstm", "all"):
        train_bilstm(args)


if __name__ == "__main__":
    main()
