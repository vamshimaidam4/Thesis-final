import { Brain, ArrowDown, Layers, GitBranch, Cpu, Database } from "lucide-react";

function Architecture() {
  const layers = [
    { name: "Input: E-commerce Review Text", color: "#dbeafe", border: "#93c5fd", icon: "📝" },
    { name: "BERT Tokenizer (WordPiece, 30K vocab)", color: "#e0e7ff", border: "#a5b4fc", icon: "🔤" },
    { name: "BERT Encoder (12 layers, 768-dim, 110M params)", color: "#c7d2fe", border: "#818cf8", icon: "🧠" },
    { name: "BiLSTM (256 hidden, bidirectional → 512-dim)", color: "#ddd6fe", border: "#a78bfa", icon: "🔄" },
    { name: "Multi-Head Attention Mechanism", color: "#ede9fe", border: "#8b5cf6", icon: "🎯" },
    { name: "Task-Specific Classification Heads", color: "#fae8ff", border: "#c084fc", icon: "📊" },
  ];

  const heads = [
    {
      name: "Document-Level Sentiment",
      desc: "Query-based attention aggregation over sentence [CLS] representations. Produces overall review sentiment through learned hierarchical importance weighting.",
      metrics: "Target: ~93% Accuracy, ~91% F1",
      color: "#10b981",
      loss: "λ_doc = 1.0"
    },
    {
      name: "Sentence-Level Sentiment",
      desc: "Per-sentence sentiment classification using individual [CLS] token representations. Captures fine-grained sentiment transitions within documents.",
      metrics: "Detects sentiment shifts across sentences",
      color: "#6366f1",
      loss: "λ_sent = 0.5"
    },
    {
      name: "Aspect Term Extraction (ATE)",
      desc: "BIO sequence tagging with CRF layer for identifying aspect terms (e.g., 'battery life', 'screen quality') from token representations.",
      metrics: "Target: F1 ~78-85% on SemEval-2014",
      color: "#f59e0b",
      loss: "λ_ate = 0.3"
    },
    {
      name: "Aspect Sentiment Classification (ASC)",
      desc: "Context-aware sentiment classification for extracted aspects. Concatenates pooled aspect representation with sentence [CLS] for prediction.",
      metrics: "Target: ~85-92% Accuracy",
      color: "#ef4444",
      loss: "λ_asc = 0.7"
    },
  ];

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Model Architecture</h2>
        <p className="page-description">HMGS: Hierarchical Multi-Granularity Sentiment Analysis System</p>
      </div>

      <div className="card card-accent">
        <h3 className="card-title"><Brain size={18} /> Architecture Overview</h3>
        <div className="architecture-diagram">
          {layers.map((layer, i) => (
            <div key={i}>
              <div
                className="arch-layer"
                style={{ background: layer.color, borderColor: layer.border }}
              >
                <span style={{ marginRight: "0.5rem" }}>{layer.icon}</span>
                {layer.name}
              </div>
              {i < layers.length - 1 && (
                <div className="arch-arrow">
                  <ArrowDown size={20} />
                </div>
              )}
            </div>
          ))}
        </div>
        <div style={{ textAlign: "center", marginTop: "1rem" }}>
          <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", maxWidth: "600px", margin: "0 auto" }}>
            Multi-task learning with shared BERT encoder. Loss = λ_doc·L_doc + λ_sent·L_sent + λ_ate·L_ate + λ_asc·L_asc
          </p>
        </div>
      </div>

      <div className="card">
        <h3 className="card-title"><GitBranch size={18} /> Classification Heads</h3>
        <div className="grid-2">
          {heads.map((head, i) => (
            <div className="result-item" key={i} style={{ borderLeft: `3px solid ${head.color}` }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.5rem" }}>
                <h4 style={{ color: head.color, fontWeight: "700" }}>{head.name}</h4>
                <span style={{ fontSize: "0.7rem", background: "#f1f5f9", padding: "0.125rem 0.5rem", borderRadius: "4px", color: "var(--text-secondary)", fontFamily: "monospace" }}>
                  {head.loss}
                </span>
              </div>
              <p style={{ fontSize: "0.85rem", marginBottom: "0.75rem", color: "var(--text-secondary)", lineHeight: "1.6" }}>{head.desc}</p>
              <p style={{ fontSize: "0.8rem", fontWeight: "600", color: "var(--text)" }}>{head.metrics}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <h3 className="card-title"><Layers size={18} /> Training Configuration</h3>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Parameter</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ["Base Model", "bert-base-uncased (110M params)"],
                  ["Hidden Dimension", "768"],
                  ["BiLSTM Hidden", "256 (bidirectional → 512)"],
                  ["Attention Heads", "12 (BERT) + 1 (Document Query)"],
                  ["CRF Tags", "3 (B-ASP, I-ASP, O)"],
                  ["BERT Learning Rate", "2e-5 (discriminative)"],
                  ["Heads Learning Rate", "1e-3"],
                  ["Weight Decay", "0.01"],
                  ["Batch Size", "16 (effective 32 w/ accumulation)"],
                  ["Max Epochs", "5"],
                  ["Early Stopping", "Patience = 2 (on val F1)"],
                  ["Optimizer", "AdamW"],
                  ["LR Scheduler", "Linear warmup (10% steps)"],
                  ["Gradient Clipping", "Max norm = 1.0"],
                  ["Max Sequence Length", "128 tokens"],
                  ["Max Doc Sentences", "10"],
                ].map(([param, value], i) => (
                  <tr key={i}>
                    <td style={{ fontWeight: "500" }}>{param}</td>
                    <td style={{ fontFamily: "monospace", fontSize: "0.8rem" }}>{value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <h3 className="card-title"><Cpu size={18} /> Multi-Task Loss Weighting</h3>
          <div style={{ padding: "1rem 0" }}>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "1.5rem", lineHeight: "1.6" }}>
              Joint training uses weighted combination of task-specific losses for balanced multi-granularity learning:
            </p>
            {[
              { name: "Document (λ_doc)", weight: 1.0, color: "#10b981", desc: "Primary objective" },
              { name: "Aspect SC (λ_asc)", weight: 0.7, color: "#ef4444", desc: "Aspect sentiment" },
              { name: "Sentence (λ_sent)", weight: 0.5, color: "#6366f1", desc: "Fine-grained signal" },
              { name: "Aspect TE (λ_ate)", weight: 0.3, color: "#f59e0b", desc: "CRF tagging loss" },
            ].map((item, i) => (
              <div key={i} style={{ marginBottom: "1rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.25rem" }}>
                  <span style={{ fontSize: "0.8rem", fontWeight: "600" }}>{item.name}</span>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>{item.desc} — {item.weight}</span>
                </div>
                <div style={{ height: "8px", background: "#e2e8f0", borderRadius: "4px", overflow: "hidden" }}>
                  <div style={{ height: "100%", width: `${item.weight * 100}%`, background: item.color, borderRadius: "4px", transition: "width 0.5s" }} />
                </div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: "1rem", padding: "1rem", background: "#f8fafc", borderRadius: "var(--radius-sm)" }}>
            <p style={{ fontSize: "0.8rem", fontFamily: "monospace", textAlign: "center", color: "var(--text)" }}>
              L = 1.0·L_doc + 0.5·L_sent + 0.3·L_ate + 0.7·L_asc
            </p>
          </div>
        </div>
      </div>

      <div className="card">
        <h3 className="card-title"><Database size={18} /> AWS Infrastructure</h3>
        <div className="grid-4">
          {[
            { service: "Amazon S3", usage: "Model checkpoint storage, versioning, and distribution", icon: "🗄️" },
            { service: "EC2 / SageMaker", usage: "GPU-accelerated training on p3.2xlarge (Tesla V100)", icon: "⚡" },
            { service: "IAM", usage: "Fine-grained access control for S3 and compute resources", icon: "🔐" },
            { service: "CloudWatch", usage: "Training metrics monitoring, alerts, and log aggregation", icon: "📈" },
          ].map((item, i) => (
            <div key={i} style={{ padding: "1.25rem", background: "#f8fafc", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", textAlign: "center" }}>
              <div style={{ fontSize: "1.5rem", marginBottom: "0.5rem" }}>{item.icon}</div>
              <div style={{ fontWeight: "700", fontSize: "0.875rem", marginBottom: "0.375rem" }}>{item.service}</div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", lineHeight: "1.5" }}>{item.usage}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3 className="card-title">Expected Performance Benchmarks</h3>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Task</th>
                <th>Accuracy</th>
                <th>F1 (Macro)</th>
                <th>Dataset</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong style={{ color: "var(--primary)" }}>HMGS</strong></td>
                <td>Document Sentiment</td>
                <td>~93%</td>
                <td>~91%</td>
                <td>Amazon Reviews (Electronics)</td>
              </tr>
              <tr>
                <td><strong style={{ color: "var(--primary)" }}>HMGS</strong></td>
                <td>Aspect Extraction</td>
                <td>-</td>
                <td>~78-85%</td>
                <td>SemEval-2014 Task 4</td>
              </tr>
              <tr>
                <td><strong style={{ color: "var(--primary)" }}>HMGS</strong></td>
                <td>Aspect Sentiment</td>
                <td>~85-92%</td>
                <td>-</td>
                <td>SemEval-2014 Task 4</td>
              </tr>
              <tr>
                <td><strong>BERT-BiLSTM</strong></td>
                <td>Document Sentiment</td>
                <td>~90%</td>
                <td>~88%</td>
                <td>Amazon Reviews (Electronics)</td>
              </tr>
              <tr>
                <td><strong>BERT-Linear</strong></td>
                <td>Document Sentiment</td>
                <td>~87%</td>
                <td>~85%</td>
                <td>Amazon Reviews (Electronics)</td>
              </tr>
              <tr>
                <td><strong>TF-IDF + LR</strong></td>
                <td>Document Sentiment</td>
                <td>~78%</td>
                <td>~76%</td>
                <td>Amazon Reviews (Electronics)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default Architecture;
