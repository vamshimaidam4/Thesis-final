import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF


class DocumentAggregator(nn.Module):
    """Query-based attention over sentence [CLS] representations."""

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
    """Hierarchical Multi-Granularity Sentiment model.
    Shared BERT encoder with 4 task heads trained jointly."""

    def __init__(self, bert_model, cfg):
        super().__init__()
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

        self.doc_agg = DocumentAggregator(h, cfg.num_sentiment_classes, cfg.classifier_dropout)

    def encode(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def forward_sent(self, cls_repr, labels=None, class_weights=None):
        logits = self.sent_head(cls_repr)
        r = {"sent_logits": logits}
        if labels is not None:
            loss_kwargs = {}
            if class_weights is not None:
                loss_kwargs["weight"] = class_weights
            r["sent_loss"] = F.cross_entropy(logits, labels, **loss_kwargs)
        return r

    def forward_ate(self, token_states, attention_mask, bio_labels=None):
        emissions = self.ate_proj(token_states)
        mask = attention_mask.bool()
        r = {"tags": self.crf.decode(emissions, mask=mask)}
        if bio_labels is not None:
            r["ate_loss"] = -self.crf(emissions, bio_labels, mask=mask, reduction="mean")
        return r

    def forward_asc(self, aspect_repr, cls_repr, labels=None, class_weights=None):
        logits = self.asc_head(torch.cat([aspect_repr, cls_repr], dim=-1))
        r = {"asc_logits": logits}
        if labels is not None:
            loss_kwargs = {}
            if class_weights is not None:
                loss_kwargs["weight"] = class_weights
            r["asc_loss"] = F.cross_entropy(logits, labels, **loss_kwargs)
        return r

    def forward_doc(self, sent_cls_batch, sent_mask, labels=None, class_weights=None):
        return self.doc_agg(sent_cls_batch, sent_mask, labels, class_weights)


class BertBiLSTMAttention(nn.Module):
    """BERT -> BiLSTM -> Attention -> Classification."""

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

    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        tok = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_out, _ = self.lstm(tok)
        sc = self.attn_w(lstm_out).squeeze(-1)
        sc = sc.masked_fill(~attention_mask.bool(), float("-inf"))
        w = F.softmax(sc, dim=-1)
        ctx = torch.bmm(w.unsqueeze(1), lstm_out).squeeze(1)
        logits = self.clf(ctx)
        r = {"logits": logits, "attention_weights": w}
        if labels is not None:
            loss_kwargs = {}
            if class_weights is not None:
                loss_kwargs["weight"] = class_weights
            r["loss"] = F.cross_entropy(logits, labels, **loss_kwargs)
        return r


class BertLinearBaseline(nn.Module):
    """BERT [CLS] -> Dropout -> Linear -> 3-class sentiment."""

    def __init__(self, bert_model, num_labels=3, dropout=0.3):
        super().__init__()
        self.bert = bert_model
        h = self.bert.config.hidden_size
        self.clf = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h, num_labels),
        )

    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logits = self.clf(cls)
        r = {"logits": logits}
        if labels is not None:
            loss_kwargs = {}
            if class_weights is not None:
                loss_kwargs["weight"] = class_weights
            r["loss"] = F.cross_entropy(logits, labels, **loss_kwargs)
        return r
