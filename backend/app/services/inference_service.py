import os
import torch
from transformers import BertTokenizerFast, BertModel
from nltk.tokenize import sent_tokenize
import nltk

from app.config import config
from app.models.sentiment_models import HMGS, BertBiLSTMAttention

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

LABEL_NAMES = ["Negative", "Neutral", "Positive"]
BIO_TAG_NAMES = ["B-ASP", "I-ASP", "O"]

DEVICE = torch.device(config.device)

_tokenizer = None
_hmgs_model = None
_bilstm_model = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = BertTokenizerFast.from_pretrained(config.model.bert_model_name)
    return _tokenizer


def load_hmgs_model():
    """Load the HMGS model from checkpoint."""
    global _hmgs_model
    checkpoint_path = os.path.join(config.model_path, "hmgs_best.pt")
    bert = BertModel.from_pretrained(config.model.bert_model_name)
    model = HMGS(bert, config.model)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"Loaded HMGS from {checkpoint_path}")
    else:
        print("No HMGS checkpoint found, using untrained model for demo")
    model.to(DEVICE)
    model.eval()
    _hmgs_model = model
    return model


def load_bilstm_model():
    """Load the BERT-BiLSTM-Attention model from checkpoint."""
    global _bilstm_model
    checkpoint_path = os.path.join(config.model_path, "bilstm_best.pt")
    bert = BertModel.from_pretrained(config.model.bert_model_name)
    model = BertBiLSTMAttention(bert, config.model.lstm_hidden, config.model.num_sentiment_classes)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"Loaded BiLSTM from {checkpoint_path}")
    else:
        print("No BiLSTM checkpoint found, using untrained model for demo")
    model.to(DEVICE)
    model.eval()
    _bilstm_model = model
    return model


def get_hmgs_model():
    if _hmgs_model is None:
        load_hmgs_model()
    return _hmgs_model


def get_bilstm_model():
    if _bilstm_model is None:
        load_bilstm_model()
    return _bilstm_model


def analyze_document_sentiment(text: str) -> dict:
    """Analyze document-level sentiment using HMGS model."""
    model = get_hmgs_model()
    tokenizer = get_tokenizer()
    cfg = config.model

    sentences = sent_tokenize(text)[:cfg.max_doc_sentences]
    cls_list = []

    for sent in sentences:
        enc = tokenizer(
            sent, max_length=cfg.max_seq_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model.bert(**enc)
            cls_list.append(out.last_hidden_state[:, 0, :])

    num_real = len(cls_list)
    while len(cls_list) < cfg.max_doc_sentences:
        cls_list.append(torch.zeros(1, cfg.hidden_dim, device=DEVICE))

    sent_cls = torch.cat(cls_list, dim=0).unsqueeze(0)
    sent_mask = torch.zeros(1, cfg.max_doc_sentences, dtype=torch.long, device=DEVICE)
    sent_mask[0, :num_real] = 1

    with torch.no_grad():
        result = model.forward_doc(sent_cls, sent_mask)

    logits = result["logits"][0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred_idx = logits.argmax().item()
    attn_weights = result["attention_weights"][0, :num_real].cpu().numpy()

    return {
        "overall_sentiment": LABEL_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {
            LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))
        },
        "sentence_attention": [
            {"sentence": sentences[i], "weight": float(attn_weights[i])}
            for i in range(num_real)
        ],
    }


def analyze_sentence_sentiment(text: str) -> dict:
    """Analyze sentence-level sentiment."""
    model = get_hmgs_model()
    tokenizer = get_tokenizer()
    cfg = config.model

    sentences = sent_tokenize(text)
    results = []

    for sent in sentences:
        enc = tokenizer(
            sent, max_length=cfg.max_seq_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model.bert(**enc)
            cls_repr = out.last_hidden_state[:, 0, :]
            sent_result = model.forward_sent(cls_repr)

        logits = sent_result["sent_logits"][0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_idx = logits.argmax().item()

        results.append({
            "sentence": sent,
            "sentiment": LABEL_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))
            },
        })

    return {"sentences": results}


def analyze_aspects(text: str) -> dict:
    """Extract aspects and their sentiments."""
    model = get_hmgs_model()
    tokenizer = get_tokenizer()
    cfg = config.model

    enc = tokenizer(
        text, max_length=cfg.max_seq_length, padding="max_length",
        truncation=True, return_tensors="pt", return_offsets_mapping=True
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        token_states = model.encode(enc["input_ids"], enc["attention_mask"])
        ate_result = model.forward_ate(token_states, enc["attention_mask"])
        cls_repr = token_states[:, 0, :]

    tags = ate_result["tags"][0]

    aspects = []
    current_aspect_tokens = []
    current_aspect_indices = []

    for idx, tag in enumerate(tags):
        if tag == 0:  # B-ASP
            if current_aspect_tokens:
                aspects.append((current_aspect_tokens, current_aspect_indices))
            current_aspect_tokens = [idx]
            current_aspect_indices = [idx]
        elif tag == 1 and current_aspect_tokens:  # I-ASP
            current_aspect_tokens.append(idx)
            current_aspect_indices.append(idx)
        else:
            if current_aspect_tokens:
                aspects.append((current_aspect_tokens, current_aspect_indices))
                current_aspect_tokens = []
                current_aspect_indices = []

    if current_aspect_tokens:
        aspects.append((current_aspect_tokens, current_aspect_indices))

    aspect_results = []
    for token_indices, _ in aspects:
        aspect_repr = token_states[0, token_indices].mean(dim=0, keepdim=True)
        with torch.no_grad():
            asc_result = model.forward_asc(aspect_repr, cls_repr)
        logits = asc_result["asc_logits"][0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_idx = logits.argmax().item()

        chars = []
        for ti in token_indices:
            if ti < len(offsets) and offsets[ti][0] != offsets[ti][1]:
                chars.append(text[offsets[ti][0]:offsets[ti][1]])
        aspect_text = " ".join(chars) if chars else f"[tokens {token_indices}]"

        aspect_results.append({
            "aspect": aspect_text,
            "sentiment": LABEL_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))
            },
        })

    return {"text": text, "aspects": aspect_results}


def analyze_full(text: str) -> dict:
    """Run all three levels of analysis."""
    doc_result = analyze_document_sentiment(text)
    sent_result = analyze_sentence_sentiment(text)
    aspect_result = analyze_aspects(text)

    return {
        "text": text,
        "document_level": doc_result,
        "sentence_level": sent_result,
        "aspect_level": aspect_result,
    }


def analyze_with_bilstm(text: str) -> dict:
    """Analyze using the BERT-BiLSTM-Attention model."""
    model = get_bilstm_model()
    tokenizer = get_tokenizer()

    enc = tokenizer(
        text, max_length=256, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    with torch.no_grad():
        result = model(**enc)

    logits = result["logits"][0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred_idx = logits.argmax().item()
    attn = result["attention_weights"][0].cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().numpy())
    mask = enc["attention_mask"][0].cpu().numpy()
    token_attention = [
        {"token": tokens[i], "weight": float(attn[i])}
        for i in range(len(tokens)) if mask[i] == 1 and tokens[i] not in ("[PAD]",)
    ]

    return {
        "sentiment": LABEL_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {
            LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))
        },
        "top_attention_tokens": sorted(token_attention, key=lambda x: -x["weight"])[:20],
    }
