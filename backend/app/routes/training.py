import os
import threading
import time
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

router = APIRouter(prefix="/api/training", tags=["training"])

# Global training state
_training_state = {
    "status": "idle",  # idle, training, completed, failed
    "model": None,
    "epoch": 0,
    "total_epochs": 0,
    "progress": 0,
    "train_loss": 0,
    "val_loss": 0,
    "val_f1": 0,
    "val_acc": 0,
    "history": [],
    "message": "",
    "started_at": None,
}


class TrainRequest(BaseModel):
    model: str = Field("hmgs", description="Model to train: hmgs, bilstm, or all")
    epochs: int = Field(3, ge=1, le=20)
    batch_size: int = Field(8, ge=2, le=64)
    train_samples: int = Field(600, ge=100, le=50000)


def _run_training(model_type: str, epochs: int, batch_size: int, train_samples: int):
    """Run training in background thread."""
    global _training_state
    import torch
    import random
    import numpy as np
    from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
    from sklearn.metrics import f1_score, accuracy_score
    from app.config import config
    from app.models.sentiment_models import HMGS, BertBiLSTMAttention

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def generate_data(n):
        set_seed(42)
        templates = {
            0: [
                "Terrible product. {} stopped working after {} days.",
                "Very disappointed with the {}. {} was awful.",
                "Do not buy this. The {} is terrible and {} is worse.",
                "Worst {} I have ever used. {} is completely broken.",
                "The {} quality is unacceptable. {} keeps failing.",
            ],
            1: [
                "The {} is okay. {} could be better but it's functional.",
                "Average {}. {} meets basic expectations but nothing more.",
                "Decent {} for the price. {} is mediocre.",
                "The {} works fine. {} is neither great nor terrible.",
                "It's an acceptable {}. {} does its job.",
            ],
            2: [
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
        for i in range(n):
            label = i % 3
            template = random.choice(templates[label])
            a1, a2 = random.sample(aspects, 2)
            text = template.format(a1, random.choice(durations)) if label == 0 else template.format(a1, a2)
            texts.append(text)
            labels.append(label)
        return texts, labels

    try:
        _training_state["status"] = "training"
        _training_state["message"] = f"Initializing {model_type.upper()} training..."
        _training_state["history"] = []

        tokenizer = BertTokenizerFast.from_pretrained(config.model.bert_model_name)
        texts, labels = generate_data(train_samples)
        split = int(len(texts) * 0.8)

        models_to_train = []
        if model_type in ("hmgs", "all"):
            models_to_train.append("hmgs")
        if model_type in ("bilstm", "all"):
            models_to_train.append("bilstm")

        _training_state["total_epochs"] = epochs * len(models_to_train)

        for current_model in models_to_train:
            _training_state["model"] = current_model
            _training_state["message"] = f"Loading BERT for {current_model.upper()}..."

            bert = BertModel.from_pretrained(config.model.bert_model_name)

            if current_model == "hmgs":
                from torch.utils.data import DataLoader, Dataset
                from nltk.tokenize import sent_tokenize
                import nltk
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)

                class _ReviewDS(Dataset):
                    def __init__(self, t, l, tok, ml=128, ms=10):
                        self.t, self.l, self.tok, self.ml, self.ms = t, l, tok, ml, ms
                    def __len__(self): return len(self.t)
                    def __getitem__(self, i):
                        sents = sent_tokenize(str(self.t[i]))[:self.ms]
                        n = len(sents)
                        ids_all, mask_all = [], []
                        for s in sents:
                            enc = self.tok(s, max_length=self.ml, padding="max_length", truncation=True, return_tensors="pt")
                            ids_all.append(enc["input_ids"].squeeze(0))
                            mask_all.append(enc["attention_mask"].squeeze(0))
                        pad = self.ms - n
                        if pad > 0:
                            ids_all.append(torch.zeros(pad, self.ml, dtype=torch.long))
                            mask_all.append(torch.zeros(pad, self.ml, dtype=torch.long))
                        input_ids = torch.cat([t.unsqueeze(0) if t.dim() == 1 else t for t in ids_all], dim=0)
                        attn_mask = torch.cat([t.unsqueeze(0) if t.dim() == 1 else t for t in mask_all], dim=0)
                        sent_mask = torch.zeros(self.ms, dtype=torch.long)
                        sent_mask[:n] = 1
                        return {"input_ids": input_ids, "attention_mask": attn_mask, "sent_mask": sent_mask,
                                "doc_label": torch.tensor(self.l[i], dtype=torch.long)}

                model = HMGS(bert, config.model).to(DEVICE)
                train_ds = _ReviewDS(texts[:split], labels[:split], tokenizer)
                val_ds = _ReviewDS(texts[split:], labels[split:], tokenizer)
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=batch_size)

                optimizer = torch.optim.AdamW([
                    {"params": model.bert.parameters(), "lr": 2e-5},
                    {"params": [p for n, p in model.named_parameters() if "bert" not in n], "lr": 1e-3},
                ], weight_decay=0.01)
                total_steps = len(train_loader) * epochs
                scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
                best_f1 = 0
                os.makedirs(config.model_path, exist_ok=True)
                h = config.model.hidden_dim

                for epoch in range(epochs):
                    _training_state["epoch"] = epoch + 1
                    _training_state["message"] = f"HMGS Epoch {epoch+1}/{epochs} - Training..."
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

                    _training_state["message"] = f"HMGS Epoch {epoch+1}/{epochs} - Validating..."
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
                    entry = {"epoch": epoch + 1, "model": "hmgs", "train_loss": round(avg_loss, 4),
                             "val_loss": round(avg_val_loss, 4), "val_f1": round(val_f1, 4), "val_acc": round(val_acc, 4)}
                    _training_state["history"].append(entry)
                    _training_state["train_loss"] = avg_loss
                    _training_state["val_loss"] = avg_val_loss
                    _training_state["val_f1"] = val_f1
                    _training_state["val_acc"] = val_acc
                    _training_state["progress"] = int(((epoch + 1) / _training_state["total_epochs"]) * 100)

                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        torch.save(model.state_dict(), os.path.join(config.model_path, "hmgs_best.pt"))

                with open(os.path.join(config.model_path, "hmgs_history.json"), "w") as f:
                    json.dump([e for e in _training_state["history"] if e["model"] == "hmgs"], f, indent=2)

            elif current_model == "bilstm":
                from torch.utils.data import DataLoader, Dataset

                class _FlatDS(Dataset):
                    def __init__(self, t, l, tok, ml=256):
                        self.t, self.l, self.tok, self.ml = t, l, tok, ml
                    def __len__(self): return len(self.t)
                    def __getitem__(self, i):
                        enc = self.tok(str(self.t[i]), max_length=self.ml, padding="max_length", truncation=True, return_tensors="pt")
                        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                                "labels": torch.tensor(self.l[i], dtype=torch.long)}

                model = BertBiLSTMAttention(bert).to(DEVICE)
                train_ds = _FlatDS(texts[:split], labels[:split], tokenizer)
                val_ds = _FlatDS(texts[split:], labels[split:], tokenizer)
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=batch_size)

                optimizer = torch.optim.AdamW([
                    {"params": model.bert.parameters(), "lr": 2e-5},
                    {"params": [p for n, p in model.named_parameters() if "bert" not in n], "lr": 1e-3},
                ], weight_decay=0.01)
                total_steps = len(train_loader) * epochs
                scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
                best_f1 = 0
                os.makedirs(config.model_path, exist_ok=True)

                for epoch in range(epochs):
                    _training_state["epoch"] = epoch + 1
                    _training_state["message"] = f"BiLSTM Epoch {epoch+1}/{epochs} - Training..."
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

                    _training_state["message"] = f"BiLSTM Epoch {epoch+1}/{epochs} - Validating..."
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
                    base_epoch = len([e for e in _training_state["history"] if e["model"] == "hmgs"])
                    entry = {"epoch": epoch + 1, "model": "bilstm", "train_loss": round(avg_loss, 4),
                             "val_loss": round(avg_val_loss, 4), "val_f1": round(val_f1, 4), "val_acc": round(val_acc, 4)}
                    _training_state["history"].append(entry)
                    _training_state["train_loss"] = avg_loss
                    _training_state["val_loss"] = avg_val_loss
                    _training_state["val_f1"] = val_f1
                    _training_state["val_acc"] = val_acc
                    _training_state["progress"] = int(((base_epoch + epoch + 1) / _training_state["total_epochs"]) * 100)

                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        torch.save(model.state_dict(), os.path.join(config.model_path, "bilstm_best.pt"))

                with open(os.path.join(config.model_path, "bilstm_history.json"), "w") as f:
                    json.dump([e for e in _training_state["history"] if e["model"] == "bilstm"], f, indent=2)

        _training_state["status"] = "completed"
        _training_state["progress"] = 100
        _training_state["message"] = "Training completed successfully!"

        # Reload models in inference service
        try:
            from app.services.inference_service import load_hmgs_model, load_bilstm_model
            if model_type in ("hmgs", "all"):
                load_hmgs_model()
            if model_type in ("bilstm", "all"):
                load_bilstm_model()
            _training_state["message"] = "Training completed! Models reloaded."
        except Exception:
            _training_state["message"] = "Training completed! Restart server to load new models."

    except Exception as e:
        _training_state["status"] = "failed"
        _training_state["message"] = f"Training failed: {str(e)}"


@router.post("/start")
def start_training(req: TrainRequest):
    """Start model training in background."""
    global _training_state
    if _training_state["status"] == "training":
        raise HTTPException(status_code=409, detail="Training already in progress")

    _training_state = {
        "status": "starting",
        "model": req.model,
        "epoch": 0,
        "total_epochs": req.epochs,
        "progress": 0,
        "train_loss": 0,
        "val_loss": 0,
        "val_f1": 0,
        "val_acc": 0,
        "history": [],
        "message": "Starting training...",
        "started_at": time.time(),
    }

    thread = threading.Thread(
        target=_run_training,
        args=(req.model, req.epochs, req.batch_size, req.train_samples),
        daemon=True,
    )
    thread.start()
    return {"message": "Training started", "model": req.model}


@router.get("/status")
def training_status():
    """Get current training status."""
    result = dict(_training_state)
    if result["started_at"]:
        result["elapsed_seconds"] = int(time.time() - result["started_at"])
    return result


@router.get("/history")
def training_history():
    """Get training history from saved files."""
    from app.config import config
    history = {}
    for name in ["hmgs", "bilstm"]:
        path = os.path.join(config.model_path, f"{name}_history.json")
        if os.path.exists(path):
            with open(path) as f:
                history[name] = json.load(f)
    return history


@router.get("/models")
def list_trained_models():
    """List available trained model files."""
    from app.config import config
    models = []
    if os.path.exists(config.model_path):
        for f in os.listdir(config.model_path):
            if f.endswith((".pt", ".pth", ".bin")):
                path = os.path.join(config.model_path, f)
                models.append({
                    "name": f,
                    "size_mb": round(os.path.getsize(path) / 1024 / 1024, 1),
                })
    return {"models": models, "model_path": config.model_path}
