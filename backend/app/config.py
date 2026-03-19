import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    bert_model_name: str = "bert-base-uncased"
    num_sentiment_classes: int = 3
    num_aspect_bio_tags: int = 3
    classifier_dropout: float = 0.3
    hidden_dim: int = 768
    max_seq_length: int = 128
    max_doc_sentences: int = 10
    lstm_hidden: int = 256


@dataclass
class TrainingConfig:
    learning_rate_bert: float = 2e-5
    learning_rate_heads: float = 1e-3
    weight_decay: float = 0.01
    max_epochs: int = 5
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 2
    lambda_doc: float = 1.0
    lambda_sent: float = 0.5
    lambda_ate: float = 0.3
    lambda_asc: float = 0.7
    train_size: int = 50_000
    val_size: int = 5_000
    test_size: int = 5_000
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


@dataclass
class AWSConfig:
    access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    region: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    s3_bucket: str = os.getenv("S3_BUCKET_NAME", "deep-learning-sentiment-models")


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    model_path: str = os.getenv("MODEL_PATH", "./models")
    device: str = os.getenv("DEVICE", "cpu")
    host: str = "0.0.0.0"
    port: int = 8000


config = AppConfig()
