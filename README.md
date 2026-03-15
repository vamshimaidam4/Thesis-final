# Hierarchical Multi-Granularity Sentiment Analysis for E-Commerce Reviews

A deep learning platform for multi-level sentiment analysis of e-commerce product reviews using BERT-based architectures with attention mechanisms and conditional random fields.

## Overview

This thesis project implements a **Hierarchical Multi-Granularity Sentiment (HMGS)** model that performs sentiment analysis at three distinct levels:

1. **Document-Level** — Overall review sentiment using query-based attention aggregation over sentence representations
2. **Sentence-Level** — Per-sentence sentiment classification with fine-grained transition detection
3. **Aspect-Level** — Aspect term extraction (ATE) with CRF tagging and aspect sentiment classification (ASC)

The system includes a full-stack web application with a React dashboard for interactive analysis and visualization.

## Architecture

```
┌─────────────────────────────────────────────┐
│           Input: Product Review Text         │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│       BERT Tokenizer (WordPiece, 30K)       │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│    BERT Encoder (12 layers, 768-dim)        │
│         Shared Feature Extraction            │
└──────────────────────┬──────────────────────┘
                       ▼
          ┌────────────┼────────────┐
          ▼            ▼            ▼
  ┌──────────────┐ ┌────────┐ ┌──────────┐
  │  Doc-Level   │ │Sent-   │ │ Aspect   │
  │  Aggregator  │ │Level   │ │ ATE+ASC  │
  │  (Attention) │ │Head    │ │ (CRF)    │
  └──────────────┘ └────────┘ └──────────┘
```

**Multi-task Loss**: `L = 1.0·L_doc + 0.5·L_sent + 0.3·L_ate + 0.7·L_asc`

## Models

| Model | Architecture | Analysis Level | Expected Performance |
|-------|-------------|---------------|---------------------|
| **HMGS** | BERT + Multi-Head Attention + CRF | Document, Sentence, Aspect | ~93% Acc, ~91% F1 |
| **BERT-BiLSTM** | BERT + BiLSTM + Attention | Document | ~90% Acc |
| **BERT-Linear** | BERT + Linear Head | Document (baseline) | ~87% Acc |
| **TF-IDF + LR** | TF-IDF + Logistic Regression | Document (baseline) | ~78% Acc |

## Tech Stack

### Backend
- **FastAPI** — REST API framework
- **PyTorch** — Deep learning framework
- **Transformers** (HuggingFace) — BERT model and tokenizer
- **pytorch-crf** — Conditional Random Field for aspect extraction
- **scikit-learn** — Evaluation metrics and baseline models
- **NLTK** — Sentence tokenization

### Frontend
- **React 19** — UI framework with Vite build tool
- **Recharts** — Data visualization (charts, radar plots, pie charts)
- **React Router** — Client-side navigation
- **Lucide React** — Icon library
- **Axios** — HTTP client

### Infrastructure
- **Docker / Docker Compose** — Containerization
- **AWS S3** — Model checkpoint storage and versioning
- **AWS EC2 / SageMaker** — GPU training (p3.2xlarge recommended)
- **GitHub Actions** — CI/CD pipeline (test, build, deploy)

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI application entry point
│   │   ├── config.py                # Configuration dataclasses
│   │   ├── models/
│   │   │   └── sentiment_models.py  # HMGS, BiLSTM, Baseline architectures
│   │   ├── routes/
│   │   │   ├── analysis.py          # Document/Sentence/Aspect/Batch endpoints
│   │   │   ├── health.py            # Health check endpoint
│   │   │   ├── aws_routes.py        # AWS S3 operations
│   │   │   └── seed_routes.py       # Seed data endpoints
│   │   └── services/
│   │       ├── inference_service.py  # Model loading & multi-level inference
│   │       ├── aws_service.py        # S3 client & operations
│   │       └── seed_service.py       # Seed data management
│   ├── tests/                        # Test suite (pytest)
│   ├── train.py                      # Training script (HMGS + BiLSTM)
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── Dashboard.jsx         # System overview & status
│       │   ├── Analyzer.jsx          # Interactive single-review analysis
│       │   ├── BatchAnalysis.jsx     # Multi-review batch processing
│       │   ├── Results.jsx           # Model evaluation & metrics
│       │   ├── SeedData.jsx          # Seed data explorer
│       │   └── Architecture.jsx      # Model architecture documentation
│       ├── components/
│       │   └── ProbabilityBars.jsx    # Sentiment probability visualization
│       └── services/
│           └── api.js                # Axios API client
├── aws/
│   └── deploy.py                     # AWS deployment utilities
├── seed_data/
│   └── sample_reviews.json           # 15 annotated e-commerce reviews
├── .github/workflows/
│   └── ci.yml                        # CI/CD pipeline
├── docker-compose.yml
├── Dockerfile
└── Enhanced_Sentiment_Analysis_Ecommerce.ipynb
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker (optional)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/vamshimaidam4/Thesis-final.git
   cd Thesis-final
   ```

2. **Backend setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
   uvicorn app.main:app --reload --port 8000
   ```

3. **Frontend setup** (new terminal)
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. **Access the application**
   - Frontend: http://localhost:5173
   - API Docs: http://localhost:8000/docs

### Docker Deployment

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

### Model Training

```bash
cd backend

# Train all models with synthetic data (quick demo)
python train.py --model all --epochs 3 --batch-size 8 --train-samples 600

# Train HMGS on larger dataset for better results
python train.py --model hmgs --epochs 5 --batch-size 16 --train-samples 50000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check with device info |
| `POST` | `/api/analysis/document` | Document-level sentiment |
| `POST` | `/api/analysis/sentence` | Sentence-level sentiment |
| `POST` | `/api/analysis/aspect` | Aspect extraction + sentiment |
| `POST` | `/api/analysis/full` | All three analysis levels |
| `POST` | `/api/analysis/batch` | Batch analysis (up to 50 reviews) |
| `GET` | `/api/seed/reviews` | List all seed reviews |
| `GET` | `/api/seed/stats` | Seed data statistics |
| `GET` | `/api/aws/status` | AWS connection status |
| `GET` | `/docs` | Interactive API documentation |

## AWS Configuration

```bash
cp .env.example .env
```

```env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=deep-learning-sentiment-models
```

```bash
# Check connectivity
python aws/deploy.py status

# Setup S3 bucket and upload models
python aws/deploy.py setup

# Download models from S3
python aws/deploy.py download
```

## Testing

```bash
cd backend
pytest tests/ -v
```

## CI/CD Pipeline

GitHub Actions workflow runs on every push to `main`:

1. **Backend Tests** — Python 3.11, pytest, flake8 linting
2. **Frontend Build** — Node 20, npm ci, vite build
3. **Docker Build** — Backend + frontend container images
4. **Deploy** — AWS credential verification, S3 model sync (main only)

## Research Notebook

`Enhanced_Sentiment_Analysis_Ecommerce.ipynb` contains the complete research pipeline:

- Data acquisition and preprocessing (Amazon Reviews 2023)
- HMGS model architecture implementation
- Two-phase training protocol (Phase 1: Doc+Sent, Phase 2: ATE+ASC)
- Evaluation framework with classification reports
- Baseline comparisons and ablation studies
- Training curves and attention weight visualizations

## License

This project is part of a Master's thesis. All rights reserved.
