"""Model definitions for the SageMaker training container.

These mirror backend/app/models/sentiment_models.py exactly so that the
state_dicts produced by training load cleanly into the inference backend.
Keep parameter names and layer shapes identical to the backend module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF


class _Cfg:
    """Lightweight stand-in for app.config.ModelConfig used by HMGS."""

    def __init__(self, classifier_dropout=0.3, num_sentiment_classes=3,
                 num_aspect_bio_tags=3):
        self.classifier_dropout = classifier_dropout
        self.num_sentiment_classes = num_sentiment_classes
        self.num_aspect_bio_tags = num_aspect_bio_tags


class DocumentAggregator(nn.Module):
    def __init__(self, hidden_dim=768, num_labels=3, dropout=0.3):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
        )

    def forward(self, sent_reprs, sent_mask, labels=None, class_weights=None):
        keys = self.key_proj(sent_reprs)
        scores = torch.einsum("bsh,h->bs", keys, self.query) / self.scale
        scores = scores.masked_fill(~sent_mask.bool(), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        doc = torch.einsum("bs,bsh->bh", attn, sent_reprs)
        logits = self.classifier(doc)
        out = {"logits": logits, "attention_weights": attn}
        if labels is not None:
            loss_kwargs = {}
            if class_weights is not None:
                loss_kwargs["weight"] = class_weights
            out["loss"] = F.cross_entropy(logits, labels, **loss_kwargs)
        return out


class HMGS(nn.Module):
    """Shared BERT + 4 task heads (document, sentence, ATE-CRF, ASC)."""

    def __init__(self, bert_model, cfg=None):
        super().__init__()
        cfg = cfg or _Cfg()
        self.bert = bert_model
        h = self.bert.config.hidden_size

        self.sent_head = nn.Sequential(
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(h // 2, cfg.num_sentiment_classes),
        )
        self.ate_proj = nn.Linear(h, cfg.num_aspect_bio_tags)
        self.crf = CRF(cfg.num_aspect_bio_tags, batch_first=True)
        self.asc_head = nn.Sequential(
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(h * 2, h // 2),
            nn.GELU(),
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(h // 2, cfg.num_sentiment_classes),
        )
        self.doc_agg = DocumentAggregator(h, cfg.num_sentiment_classes,
                                          cfg.classifier_dropout)

    def encode(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def forward_sent(self, cls_repr, labels=None):
        logits = self.sent_head(cls_repr)
        out = {"sent_logits": logits}
        if labels is not None:
            out["sent_loss"] = F.cross_entropy(logits, labels)
        return out

    def forward_doc(self, sent_reprs, sent_mask, labels=None):
        return self.doc_agg(sent_reprs, sent_mask, labels)


class BertBiLSTMAttention(nn.Module):
    """BERT -> BiLSTM -> Attention -> Classification (mirrors backend)."""

    def __init__(self, bert_model, lstm_h=256, num_labels=3, dropout=0.3):
        super().__init__()
        self.bert = bert_model
        h = self.bert.config.hidden_size
        self.lstm = nn.LSTM(h, lstm_h, batch_first=True, bidirectional=True)
        dim = lstm_h * 2
        self.attn_w = nn.Linear(dim, 1)
        self.clf = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_labels),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        tok = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_out, _ = self.lstm(tok)
        sc = self.attn_w(lstm_out).squeeze(-1)
        sc = sc.masked_fill(~attention_mask.bool(), float("-inf"))
        w = F.softmax(sc, dim=-1)
        ctx = torch.bmm(w.unsqueeze(1), lstm_out).squeeze(1)
        logits = self.clf(ctx)
        r = {"logits": logits, "attention_weights": w}
        if labels is not None:
            r["loss"] = F.cross_entropy(logits, labels)
        return r
