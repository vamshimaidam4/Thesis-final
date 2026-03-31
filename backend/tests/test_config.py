import pytest
from app.config import ModelConfig, TrainingConfig, AWSConfig, AppConfig


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.bert_model_name == "bert-base-uncased"
    assert cfg.num_sentiment_classes == 3
    assert cfg.num_aspect_bio_tags == 3
    assert cfg.classifier_dropout == 0.3
    assert cfg.hidden_dim == 768
    assert cfg.max_seq_length == 128
    assert cfg.max_doc_sentences == 10
    assert cfg.lstm_hidden == 256


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.learning_rate_bert == 2e-5
    assert cfg.learning_rate_heads == 1e-3
    assert cfg.max_epochs == 5
    assert cfg.batch_size == 16
    assert cfg.early_stopping_patience == 2
    assert cfg.train_size == 50_000
    assert len(cfg.seeds) == 3


def test_training_config_loss_weights():
    cfg = TrainingConfig()
    assert cfg.lambda_doc == 1.0
    assert cfg.lambda_sent == 0.5
    assert cfg.lambda_ate == 0.3
    assert cfg.lambda_asc == 0.7


def test_app_config_structure():
    cfg = AppConfig()
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.aws, AWSConfig)
    assert cfg.port == 8000
