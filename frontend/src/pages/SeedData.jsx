import { useState, useEffect } from "react";
import { getSeedReviews, analyzeFull } from "../services/api";
import ProbabilityBars from "../components/ProbabilityBars";
import { Database, Star, Search } from "lucide-react";

function SeedData() {
  const [reviews, setReviews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(null);
  const [results, setResults] = useState({});
  const [filter, setFilter] = useState("all");

  useEffect(() => {
    getSeedReviews()
      .then((data) => setReviews(data.reviews || []))
      .catch(() => setReviews([]))
      .finally(() => setLoading(false));
  }, []);

  const handleAnalyze = async (review) => {
    setAnalyzing(review.id);
    try {
      const data = await analyzeFull(review.text);
      setResults((prev) => ({ ...prev, [review.id]: data }));
    } catch {
      setResults((prev) => ({ ...prev, [review.id]: { error: true } }));
    }
    setAnalyzing(null);
  };

  const analyzeAll = async () => {
    for (const review of reviews) {
      if (!results[review.id]) {
        await handleAnalyze(review);
      }
    }
  };

  const filteredReviews = filter === "all" ? reviews : reviews.filter(r => r.expected_sentiment === filter);

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner" />
        Loading seed data...
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Seed Data Explorer</h2>
        <p className="page-description">15 annotated e-commerce reviews for model validation and testing</p>
      </div>

      <div className="card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "1rem 1.5rem" }}>
        <div className="btn-group" style={{ marginBottom: 0 }}>
          <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginRight: "0.5rem" }}>Filter:</span>
          {["all", "Positive", "Neutral", "Negative"].map(f => (
            <button key={f} className={`btn btn-sm ${filter === f ? "btn-primary" : "btn-secondary"}`} onClick={() => setFilter(f)}>
              {f === "all" ? `All (${reviews.length})` : `${f} (${reviews.filter(r => r.expected_sentiment === f).length})`}
            </button>
          ))}
        </div>
        <button className="btn btn-primary btn-sm" onClick={analyzeAll}>
          <Search size={14} /> Analyze All
        </button>
      </div>

      {filteredReviews.map((review) => {
        const res = results[review.id];
        const pred = res?.document_level?.overall_sentiment;
        const isCorrect = pred && pred === review.expected_sentiment;
        const isWrong = pred && pred !== review.expected_sentiment;

        return (
          <div className="card" key={review.id} style={{
            borderLeft: isCorrect ? "3px solid var(--positive)" : isWrong ? "3px solid var(--negative)" : "3px solid transparent"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.5rem" }}>
                  <span style={{ fontWeight: "700", color: "var(--primary)", fontSize: "0.875rem" }}>#{review.id}</span>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", background: "#f1f5f9", padding: "0.125rem 0.5rem", borderRadius: "4px" }}>{review.category}</span>
                  <span style={{ display: "flex", alignItems: "center", gap: "0.125rem" }}>
                    {Array.from({ length: 5 }, (_, i) => (
                      <Star key={i} size={12} fill={i < review.star_rating ? "#f59e0b" : "none"} color={i < review.star_rating ? "#f59e0b" : "#cbd5e1"} />
                    ))}
                  </span>
                </div>
                <p style={{ fontSize: "0.9rem", marginBottom: "0.75rem", lineHeight: "1.6" }}>"{review.text}"</p>
                <div style={{ display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap" }}>
                  <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                    Expected: <span className={`sentiment-badge sentiment-${review.expected_sentiment}`}>{review.expected_sentiment}</span>
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                    Aspects: {review.expected_aspects?.map((a, i) => (
                      <span key={i} style={{ background: "#f1f5f9", padding: "0.1rem 0.5rem", borderRadius: "4px", marginLeft: "0.25rem", fontSize: "0.7rem" }}>{a}</span>
                    ))}
                  </div>
                </div>
              </div>
              <button
                className="btn btn-primary btn-sm"
                onClick={() => handleAnalyze(review)}
                disabled={analyzing === review.id}
                style={{ marginLeft: "1rem", whiteSpace: "nowrap" }}
              >
                {analyzing === review.id ? <><div className="spinner" style={{ width: "14px", height: "14px", margin: 0, marginRight: "0.375rem" }} /> Analyzing</> : "Analyze"}
              </button>
            </div>

            {res && !res.error && (
              <div style={{ marginTop: "1rem", padding: "1rem", background: isCorrect ? "rgba(16, 185, 129, 0.04)" : isWrong ? "rgba(239, 68, 68, 0.04)" : "#f8fafc", borderRadius: "var(--radius-sm)", border: `1px solid ${isCorrect ? "rgba(16, 185, 129, 0.2)" : isWrong ? "rgba(239, 68, 68, 0.2)" : "var(--border)"}` }}>
                <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "0.5rem" }}>
                  <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>Prediction:</span>
                  <span className={`sentiment-badge sentiment-${pred}`}>{pred}</span>
                  <span style={{ fontSize: "0.8rem", fontWeight: "600" }}>
                    {((res.document_level?.confidence || 0) * 100).toFixed(1)}%
                  </span>
                  <span style={{ fontSize: "0.9rem" }}>{isCorrect ? "✓ Correct" : "✗ Incorrect"}</span>
                </div>
                {res.aspect_level?.aspects?.length > 0 && (
                  <div style={{ marginTop: "0.5rem" }}>
                    <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>Detected Aspects: </span>
                    {res.aspect_level.aspects.map((a, i) => (
                      <span key={i} className={`sentiment-badge sentiment-${a.sentiment}`} style={{ margin: "0 2px", fontSize: "0.7rem", padding: "0.1rem 0.5rem" }}>
                        {a.aspect}: {a.sentiment}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
            {res?.error && (
              <div className="error" style={{ marginTop: "0.75rem" }}>Analysis failed. Is the backend running?</div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default SeedData;
