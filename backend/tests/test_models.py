import pytest
import torch
from app.config import ModelConfig
from app.models.sentiment_models import (
    DocumentAggregator,
    HMGS,
    BertBiLSTMAttention,
    BertLinearBaseline,
)


@pytest.fixture
def model_config():
    return ModelConfig()


class TestDocumentAggregator:
    def test_forward_shape(self):
        agg = DocumentAggregator(hidden_dim=64, num_labels=3, dropout=0.1)
        sent_reprs = torch.randn(2, 5, 64)
        sent_mask = torch.ones(2, 5, dtype=torch.long)
        result = agg(sent_reprs, sent_mask)
        assert result["logits"].shape == (2, 3)
        assert result["attention_weights"].shape == (2, 5)

    def test_forward_with_labels(self):
        agg = DocumentAggregator(hidden_dim=64, num_labels=3, dropout=0.1)
        sent_reprs = torch.randn(2, 5, 64)
        sent_mask = torch.ones(2, 5, dtype=torch.long)
        labels = torch.tensor([0, 2])
        result = agg(sent_reprs, sent_mask, labels)
        assert "loss" in result
        assert result["loss"].dim() == 0

    def test_attention_mask(self):
        agg = DocumentAggregator(hidden_dim=64, num_labels=3, dropout=0.1)
        sent_reprs = torch.randn(1, 5, 64)
        sent_mask = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.long)
        result = agg(sent_reprs, sent_mask)
        weights = result["attention_weights"][0]
        assert weights[2].item() < 0.01
        assert weights[3].item() < 0.01

    def test_attention_weights_sum_to_one(self):
        agg = DocumentAggregator(hidden_dim=64, num_labels=3, dropout=0.1)
        sent_reprs = torch.randn(1, 5, 64)
        sent_mask = torch.ones(1, 5, dtype=torch.long)
        result = agg(sent_reprs, sent_mask)
        weights_sum = result["attention_weights"][0].sum().item()
        assert abs(weights_sum - 1.0) < 1e-5


class TestBertBiLSTMAttention:
    def test_output_keys(self):
        from transformers import BertModel, BertConfig
        bert_cfg = BertConfig(hidden_size=64, num_hidden_layers=1,
                              num_attention_heads=1, intermediate_size=128,
                              vocab_size=100)
        bert = BertModel(bert_cfg)
        model = BertBiLSTMAttention(bert, lstm_h=32, num_labels=3, dropout=0.1)

        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)
        result = model(input_ids, attention_mask)

        assert "logits" in result
        assert "attention_weights" in result
        assert result["logits"].shape == (2, 3)

    def test_with_labels(self):
        from transformers import BertModel, BertConfig
        bert_cfg = BertConfig(hidden_size=64, num_hidden_layers=1,
                              num_attention_heads=1, intermediate_size=128,
                              vocab_size=100)
        bert = BertModel(bert_cfg)
        model = BertBiLSTMAttention(bert, lstm_h=32, num_labels=3, dropout=0.1)

        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)
        labels = torch.tensor([1, 2])
        result = model(input_ids, attention_mask, labels)

        assert "loss" in result
        assert result["loss"].requires_grad


class TestBertLinearBaseline:
    def test_output_shape(self):
        from transformers import BertModel, BertConfig
        bert_cfg = BertConfig(hidden_size=64, num_hidden_layers=1,
                              num_attention_heads=1, intermediate_size=128,
                              vocab_size=100)
        bert = BertModel(bert_cfg)
        model = BertLinearBaseline(bert, num_labels=3, dropout=0.1)

        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)
        result = model(input_ids, attention_mask)

        assert result["logits"].shape == (2, 3)
