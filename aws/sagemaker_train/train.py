"""SageMaker training entrypoint.

Pulls the McAuley-Lab/Amazon-Reviews-2023 dataset from HuggingFace, maps
1-5 star ratings to 3-class sentiment (Negative/Neutral/Positive), and
trains the HMGS and BiLSTM models. Saves checkpoints to /opt/ml/model
which SageMaker tars and uploads to S3 automatically.

Hyperparameters (passed by SageMaker as CLI args):
  --model              hmgs | bilstm | both
  --epochs             default 3
  --batch-size         default 16
  --train-samples      default 10000
  --val-samples        default 1000
  --hf-config          subset of Amazon Reviews 2023 (default raw_review_All_Beauty)
  --max-len            default 128
"""
import argparse
import json
import os
import random
import sys
import time

import nltk
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (BertModel, BertTokenizerFast,
                          get_linear_schedule_with_warmup)

# Local module (lives next to this file in the SageMaker source dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import HMGS, BertBiLSTMAttention  # noqa: E402

LABELS = ["Negative", "Neutral", "Positive"]
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "./model_out")
SM_OUTPUT_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", SM_MODEL_DIR)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stars_to_label(rating):
    """Map a 1-5 star rating to {0=Neg, 1=Neu, 2=Pos}."""
    r = float(rating)
    if r <= 2:
        return 0
    if r >= 4:
        return 2
    return 1


def fetch_amazon_reviews(hf_config, n_total, seed=42):
    """Stream Amazon Reviews 2023 from HuggingFace and balance classes."""
    print(f"[data] streaming {hf_config} from HuggingFace...", flush=True)
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        hf_config,
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    per_class = n_total // 3
    buckets = {0: [], 1: [], 2: []}
    seen = 0
    for row in ds:
        seen += 1
        text = (row.get("title") or "") + ". " + (row.get("text") or "")
        text = text.strip(". ").strip()
        if len(text) < 10:
            continue
        rating = row.get("rating")
        if rating is None:
            continue
        label = stars_to_label(rating)
        if len(buckets[label]) < per_class:
            buckets[label].append(text)
        if all(len(v) >= per_class for v in buckets.values()):
            break
        if seen >= n_total * 50:  # safety: abort if we cannot balance
            break

    texts, labels = [], []
    for lab, items in buckets.items():
        for t in items:
            texts.append(t)
            labels.append(lab)

    rng = random.Random(seed)
    paired = list(zip(texts, labels))
    rng.shuffle(paired)
    texts = [t for t, _ in paired]
    labels = [lab for _, lab in paired]
    print(f"[data] collected {len(texts)} reviews "
          f"(class counts: {[len(buckets[c]) for c in (0,1,2)]})", flush=True)
    return texts, labels


class FlatDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts, self.labels, self.tok, self.max_len = texts, labels, tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], max_length=self.max_len,
                       padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }


class SentenceDataset(Dataset):
    """Splits each review into sentences for the HMGS doc aggregator."""

    def __init__(self, texts, labels, tokenizer, max_len=128, max_sents=8):
        self.texts, self.labels, self.tok = texts, labels, tokenizer
        self.max_len, self.max_sents = max_len, max_sents

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        sents = nltk.sent_tokenize(self.texts[i])[:self.max_sents]
        n = len(sents) or 1
        if not sents:
            sents = [self.texts[i][:200]]
        ids, masks = [], []
        for s in sents:
            enc = self.tok(s, max_length=self.max_len, padding="max_length",
                           truncation=True, return_tensors="pt")
            ids.append(enc["input_ids"].squeeze(0))
            masks.append(enc["attention_mask"].squeeze(0))
        while len(ids) < self.max_sents:
            ids.append(torch.zeros(self.max_len, dtype=torch.long))
            masks.append(torch.zeros(self.max_len, dtype=torch.long))
        sent_mask = torch.zeros(self.max_sents, dtype=torch.long)
        sent_mask[:n] = 1
        return {
            "input_ids": torch.stack(ids, 0),
            "attention_mask": torch.stack(masks, 0),
            "sent_mask": sent_mask,
            "label": torch.tensor(self.labels[i], dtype=torch.long),
        }


def evaluate_flat(model, loader, device):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            lab = b["labels"].to(device)
            r = model(ids, mask, lab)
            preds.extend(r["logits"].argmax(-1).cpu().numpy())
            true.extend(lab.cpu().numpy())
    return {
        "accuracy": float(accuracy_score(true, preds)),
        "macro_f1": float(f1_score(true, preds, average="macro")),
        "report": classification_report(true, preds, target_names=LABELS,
                                        zero_division=0, digits=4),
    }


def evaluate_hmgs(model, loader, device):
    model.eval()
    preds, true = [], []
    h = model.bert.config.hidden_size
    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            sm = b["sent_mask"].to(device)
            lab = b["label"].to(device)
            B, S, L = ids.shape
            flat_ids = ids.view(B * S, L)
            flat_mask = mask.view(B * S, L)
            real = sm.view(B * S).bool()
            cls = torch.zeros(B * S, h, device=device)
            if real.any():
                out = model.bert(input_ids=flat_ids[real], attention_mask=flat_mask[real])
                cls[real] = out.last_hidden_state[:, 0, :]
            r = model.forward_doc(cls.view(B, S, -1), sm)
            preds.extend(r["logits"].argmax(-1).cpu().numpy())
            true.extend(lab.cpu().numpy())
    return {
        "accuracy": float(accuracy_score(true, preds)),
        "macro_f1": float(f1_score(true, preds, average="macro")),
        "report": classification_report(true, preds, target_names=LABELS,
                                        zero_division=0, digits=4),
    }


def train_bilstm(texts_tr, lbl_tr, texts_va, lbl_va, tokenizer, args, device):
    print("[bilstm] starting training", flush=True)
    bert = BertModel.from_pretrained("bert-base-uncased")
    model = BertBiLSTMAttention(bert).to(device)

    train_loader = DataLoader(FlatDataset(texts_tr, lbl_tr, tokenizer, args.max_len),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FlatDataset(texts_va, lbl_va, tokenizer, args.max_len),
                            batch_size=args.batch_size)

    opt = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": 2e-5},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("bert.")], "lr": 1e-3},
    ], weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * 0.1), total_steps)

    history, best_f1 = [], 0.0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for b in train_loader:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            lab = b["labels"].to(device)
            r = model(ids, mask, lab)
            r["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += r["loss"].item()
        train_loss = running / len(train_loader)
        metrics = evaluate_flat(model, val_loader, device)
        history.append({"epoch": ep + 1, "train_loss": train_loss, **{k: v for k, v in metrics.items() if k != "report"}})
        print(f"[bilstm] epoch {ep+1}/{args.epochs} loss={train_loss:.4f} "
              f"acc={metrics['accuracy']:.4f} f1={metrics['macro_f1']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), os.path.join(SM_MODEL_DIR, "bilstm_best.pt"))
            with open(os.path.join(SM_MODEL_DIR, "bilstm_report.txt"), "w") as f:
                f.write(metrics["report"])
    with open(os.path.join(SM_MODEL_DIR, "bilstm_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[bilstm] done. best macro-F1 = {best_f1:.4f}", flush=True)
    return best_f1


def train_hmgs(texts_tr, lbl_tr, texts_va, lbl_va, tokenizer, args, device):
    print("[hmgs] starting training", flush=True)
    bert = BertModel.from_pretrained("bert-base-uncased")
    model = HMGS(bert).to(device)

    train_loader = DataLoader(SentenceDataset(texts_tr, lbl_tr, tokenizer, args.max_len),
                              batch_size=max(args.batch_size // 2, 4), shuffle=True)
    val_loader = DataLoader(SentenceDataset(texts_va, lbl_va, tokenizer, args.max_len),
                            batch_size=max(args.batch_size // 2, 4))

    opt = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": 2e-5},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("bert.")], "lr": 1e-3},
    ], weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * 0.1), total_steps)
    h = bert.config.hidden_size

    history, best_f1 = [], 0.0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for b in train_loader:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            sm = b["sent_mask"].to(device)
            lab = b["label"].to(device)
            B, S, L = ids.shape
            flat_ids = ids.view(B * S, L)
            flat_mask = mask.view(B * S, L)
            real = sm.view(B * S).bool()
            cls = torch.zeros(B * S, h, device=device)
            if real.any():
                out = model.bert(input_ids=flat_ids[real], attention_mask=flat_mask[real])
                cls[real] = out.last_hidden_state[:, 0, :]
            r = model.forward_doc(cls.view(B, S, -1), sm, lab)
            r["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += r["loss"].item()
        train_loss = running / len(train_loader)
        metrics = evaluate_hmgs(model, val_loader, device)
        history.append({"epoch": ep + 1, "train_loss": train_loss, **{k: v for k, v in metrics.items() if k != "report"}})
        print(f"[hmgs] epoch {ep+1}/{args.epochs} loss={train_loss:.4f} "
              f"acc={metrics['accuracy']:.4f} f1={metrics['macro_f1']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), os.path.join(SM_MODEL_DIR, "hmgs_best.pt"))
            with open(os.path.join(SM_MODEL_DIR, "hmgs_report.txt"), "w") as f:
                f.write(metrics["report"])
    with open(os.path.join(SM_MODEL_DIR, "hmgs_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[hmgs] done. best macro-F1 = {best_f1:.4f}", flush=True)
    return best_f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["hmgs", "bilstm", "both"], default="both")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-samples", type=int, default=10000)
    p.add_argument("--val-samples", type=int, default=1000)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--hf-config", default="raw_review_All_Beauty")
    args = p.parse_args()

    os.makedirs(SM_MODEL_DIR, exist_ok=True)
    set_seed(42)

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} torch={torch.__version__}", flush=True)
    if torch.cuda.is_available():
        print(f"[setup] gpu={torch.cuda.get_device_name(0)}", flush=True)

    texts, labels = fetch_amazon_reviews(args.hf_config,
                                         args.train_samples + args.val_samples)
    cut = args.train_samples
    texts_tr, lbl_tr = texts[:cut], labels[:cut]
    texts_va, lbl_va = texts[cut:cut + args.val_samples], labels[cut:cut + args.val_samples]
    print(f"[data] train={len(texts_tr)} val={len(texts_va)}", flush=True)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    summary = {"args": vars(args)}
    if args.model in ("bilstm", "both"):
        summary["bilstm_best_f1"] = train_bilstm(
            texts_tr, lbl_tr, texts_va, lbl_va, tokenizer, args, device)
    if args.model in ("hmgs", "both"):
        summary["hmgs_best_f1"] = train_hmgs(
            texts_tr, lbl_tr, texts_va, lbl_va, tokenizer, args, device)

    with open(os.path.join(SM_MODEL_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] artifacts saved to {SM_MODEL_DIR}", flush=True)
    print(f"[done] summary: {json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
