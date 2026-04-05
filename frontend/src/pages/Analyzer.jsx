import { useState } from "react";
import { analyzeFull } from "../services/api";
import ProbabilityBars from "../components/ProbabilityBars";
import { FlaskConical, FileText, MessageSquare, Tags, Sparkles } from "lucide-react";

const SAMPLE_REVIEWS = [
  { label: "Positive", text: "This laptop is absolutely amazing! The battery life lasts all day and the screen quality is stunning. Best purchase I've made this year." },
  { label: "Negative", text: "Terrible product. The keyboard stopped working after just two weeks. Customer service was unhelpful and rude. Complete waste of money." },
  { label: "Mixed", text: "The camera quality is great but the battery drains too fast. Mixed feelings overall. The design is sleek but the software is buggy." },
  { label: "Positive", text: "Fast shipping and the product was exactly as described. The sound quality is incredible for the price. Would definitely buy again." },
];

function Analyzer() {
  const [text, setText] = useState("");
  const [model, setModel] = useState("hmgs");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeFull(text, model);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
    setLoading(false);
  };

  const getSentimentColor = (sentiment) => {
    if (sentiment === "Positive") return "#10b981";
    if (sentiment === "Negative") return "#ef4444";
    return "#f59e0b";
  };

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Sentiment Analyzer</h2>
        <p className="page-description">Analyze e-commerce reviews at document, sentence, and aspect levels</p>
      </div>

      <div className="card card-accent">
        <h3 className="card-title"><FileText size={18} /> Enter Review Text</h3>
        <div className="textarea-wrapper">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type or paste a product review here for multi-level sentiment analysis..."
          />
        </div>
        <div className="btn-group">
          <div className="select-wrapper">
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="hmgs">HMGS (Multi-Level)</option>
              <option value="bilstm">BERT-BiLSTM-Attention</option>
            </select>
          </div>
          <button className="btn btn-primary" onClick={handleAnalyze} disabled={loading || !text.trim()}>
            <Sparkles size={16} />
            {loading ? "Analyzing..." : "Analyze Sentiment"}
          </button>
          {text && <button className="btn btn-secondary btn-sm" onClick={() => { setText(""); setResult(null); }}>Clear</button>}
        </div>
        <div style={{ marginTop: "0.5rem" }}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginRight: "0.5rem" }}>Try a sample:</span>
          <div className="btn-group" style={{ marginTop: "0.375rem" }}>
            {SAMPLE_REVIEWS.map((review, i) => (
              <button key={i} className="btn btn-secondary btn-sm" onClick={() => setText(review.text)}>
                <span className={`sentiment-badge sentiment-${review.label === "Mixed" ? "Neutral" : review.label}`} style={{ fontSize: "0.7rem", padding: "0.1rem 0.5rem" }}>
                  {review.label}
                </span>
                Sample {i + 1}
              </button>
            ))}
          </div>
        </div>
      </div>

      {loading && (
        <div className="loading" style={{ flexDirection: "column", gap: "0.75rem" }}>
          <div className="spinner" />
          <span>Running {model === "hmgs" ? "multi-level" : "BiLSTM"} sentiment analysis...</span>
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {result && !loading && (
        <div className="results-section">
          {/* Document Level */}
          {result.document_level && (
            <div className="card">
              <h3 className="card-title"><FileText size={18} /> Document-Level Sentiment</h3>
              <div style={{ display: "flex", alignItems: "center", gap: "1.5rem", marginBottom: "1rem" }}>
                <div style={{
                  width: "80px", height: "80px", borderRadius: "50%",
                  border: `4px solid ${getSentimentColor(result.document_level.overall_sentiment)}`,
                  display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                  background: "#fafbfc"
                }}>
                  <span style={{ fontSize: "1rem", fontWeight: "800" }}>{(result.document_level.confidence * 100).toFixed(0)}%</span>
                  <span style={{ fontSize: "0.55rem", color: "var(--text-secondary)" }}>confidence</span>
                </div>
                <div>
                  <span className={`sentiment-badge sentiment-${result.document_level.overall_sentiment}`} style={{ fontSize: "1rem", padding: "0.375rem 1.25rem" }}>
                    {result.document_level.overall_sentiment}
                  </span>
                  <p style={{ marginTop: "0.5rem", fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                    Overall document sentiment based on hierarchical attention aggregation
                  </p>
                </div>
              </div>
              <ProbabilityBars probabilities={result.document_level.probabilities} />

              {result.document_level.sentence_attention?.length > 0 && (
                <div style={{ marginTop: "1.5rem" }}>
                  <h4 style={{ fontSize: "0.85rem", fontWeight: "600", marginBottom: "0.75rem", color: "var(--text-secondary)" }}>
                    Sentence Attention Weights
                  </h4>
                  {result.document_level.sentence_attention.map((sa, i) => {
                    const maxW = Math.max(...result.document_level.sentence_attention.map(s => s.weight));
                    return (
                      <div className="attention-bar" key={i}>
                        <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", width: "20px" }}>S{i + 1}</span>
                        <span className="attention-text">{sa.sentence}</span>
                        <div className="attention-weight">
                          <div className="attention-fill" style={{ width: `${(sa.weight / maxW) * 100}%` }} />
                        </div>
                        <span style={{ fontSize: "0.75rem", width: "45px", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: "600" }}>
                          {(sa.weight * 100).toFixed(1)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* Sentence Level */}
          {result.sentence_level && (
            <div className="card">
              <h3 className="card-title"><MessageSquare size={18} /> Sentence-Level Analysis</h3>
              {result.sentence_level.sentences.map((s, i) => (
                <div className="result-item" key={i}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.5rem" }}>
                    <p className="sentence">
                      <span style={{ color: "var(--text-muted)", fontStyle: "normal", marginRight: "0.5rem" }}>S{i + 1}.</span>
                      "{s.sentence}"
                    </p>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexShrink: 0 }}>
                      <span className={`sentiment-badge sentiment-${s.sentiment}`}>{s.sentiment}</span>
                      <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)", fontWeight: "600" }}>
                        {(s.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <ProbabilityBars probabilities={s.probabilities} />
                </div>
              ))}
            </div>
          )}

          {/* Aspect Level */}
          {result.aspect_level && (
            <div className="card">
              <h3 className="card-title"><Tags size={18} /> Aspect-Level Analysis</h3>
              {result.aspect_level.aspects.length > 0 ? (
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                  {result.aspect_level.aspects.map((a, i) => (
                    <div className="aspect-chip" key={i}>
                      <strong style={{ color: "var(--text)" }}>{a.aspect}</strong>
                      <span className={`sentiment-badge sentiment-${a.sentiment}`} style={{ fontSize: "0.7rem", padding: "0.125rem 0.5rem" }}>
                        {a.sentiment}
                      </span>
                      <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: "600" }}>
                        {(a.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ color: "var(--text-secondary)", fontStyle: "italic" }}>No specific aspects detected in this text.</p>
              )}
            </div>
          )}

          {/* BiLSTM Results */}
          {result.bilstm_analysis && (
            <div className="card">
              <h3 className="card-title"><Sparkles size={18} /> BiLSTM-Attention Analysis</h3>
              <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1rem" }}>
                <span className={`sentiment-badge sentiment-${result.bilstm_analysis.sentiment}`} style={{ fontSize: "1rem", padding: "0.375rem 1.25rem" }}>
                  {result.bilstm_analysis.sentiment}
                </span>
                <span style={{ color: "var(--text-secondary)", fontWeight: "600" }}>
                  {(result.bilstm_analysis.confidence * 100).toFixed(1)}% confidence
                </span>
              </div>
              <ProbabilityBars probabilities={result.bilstm_analysis.probabilities} />
              {result.bilstm_analysis.top_attention_tokens && (
                <div style={{ marginTop: "1.5rem" }}>
                  <h4 style={{ fontSize: "0.85rem", fontWeight: "600", marginBottom: "0.75rem", color: "var(--text-secondary)" }}>
                    Top Attention Tokens
                  </h4>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "0.375rem" }}>
                    {result.bilstm_analysis.top_attention_tokens.slice(0, 15).map((t, i) => (
                      <span
                        key={i}
                        style={{
                          padding: "0.25rem 0.625rem",
                          background: `rgba(99, 102, 241, ${Math.min(t.weight * 10, 0.85)})`,
                          color: t.weight * 10 > 0.4 ? "white" : "var(--text)",
                          borderRadius: "6px",
                          fontSize: "0.8rem",
                          fontWeight: "500",
                        }}
                      >
                        {t.token}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default Analyzer;
