import { useState, useEffect } from "react";
import { getSeedReviews, analyzeFull, analyzeBatch } from "../services/api";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart, Pie, Cell, ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, LineChart, Line } from "recharts";
import ProbabilityBars from "../components/ProbabilityBars";

const COLORS = { Positive: "#10b981", Neutral: "#f59e0b", Negative: "#ef4444" };

function Results() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");
  const [error, setError] = useState(null);

  const runFullEvaluation = async () => {
    setLoading(true);
    setError(null);
    try {
      const seedData = await getSeedReviews();
      const reviews = seedData.reviews || [];
      const texts = reviews.map(r => r.text);

      const batchResult = await analyzeBatch(texts);

      const combined = reviews.map((review, i) => ({
        ...review,
        prediction: batchResult.results[i],
      }));

      const stats = computeStats(combined);
      setResults({ reviews: combined, stats, rawBatch: batchResult });
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Failed to run evaluation");
    }
    setLoading(false);
  };

  const computeStats = (data) => {
    let correct = 0;
    let total = data.length;
    const confusionMatrix = { Positive: { Positive: 0, Neutral: 0, Negative: 0 }, Neutral: { Positive: 0, Neutral: 0, Negative: 0 }, Negative: { Positive: 0, Neutral: 0, Negative: 0 } };
    const confidences = [];
    const sentimentDist = { Positive: 0, Neutral: 0, Negative: 0 };
    const predDist = { Positive: 0, Neutral: 0, Negative: 0 };

    data.forEach(item => {
      const expected = item.expected_sentiment;
      const predicted = item.prediction?.document_level?.overall_sentiment;
      const confidence = item.prediction?.document_level?.confidence || 0;

      if (expected && predicted) {
        if (expected === predicted) correct++;
        confusionMatrix[expected][predicted]++;
        sentimentDist[expected]++;
        predDist[predicted]++;
        confidences.push({ id: item.id, confidence, correct: expected === predicted });
      }
    });

    const accuracy = total > 0 ? correct / total : 0;
    const avgConfidence = confidences.length > 0 ? confidences.reduce((s, c) => s + c.confidence, 0) / confidences.length : 0;

    const perClass = {};
    ["Positive", "Neutral", "Negative"].forEach(cls => {
      const tp = confusionMatrix[cls][cls];
      const fp = Object.values(confusionMatrix).reduce((s, row) => s + (row[cls] || 0), 0) - tp;
      const fn = Object.values(confusionMatrix[cls]).reduce((s, v) => s + v, 0) - tp;
      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
      perClass[cls] = { precision, recall, f1, tp, fp, fn };
    });

    const macroF1 = Object.values(perClass).reduce((s, c) => s + c.f1, 0) / 3;

    return { accuracy, correct, total, confusionMatrix, confidences, sentimentDist, predDist, perClass, macroF1, avgConfidence };
  };

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Model Evaluation Results</h2>
        <p className="page-description">Run the HMGS model against seed data and view comprehensive performance metrics</p>
      </div>

      {!results && !loading && (
        <div className="card card-accent" style={{ textAlign: "center", padding: "3rem" }}>
          <h3 style={{ fontSize: "1.25rem", marginBottom: "0.75rem" }}>Ready to Evaluate</h3>
          <p style={{ color: "var(--text-secondary)", marginBottom: "1.5rem", maxWidth: "500px", margin: "0 auto 1.5rem" }}>
            This will analyze all 15 seed reviews using the HMGS model and generate detailed performance metrics including accuracy, F1-score, confusion matrix, and per-class statistics.
          </p>
          <button className="btn btn-primary" onClick={runFullEvaluation} style={{ padding: "0.75rem 2.5rem", fontSize: "1rem" }}>
            Run Full Evaluation
          </button>
        </div>
      )}

      {loading && (
        <div className="loading" style={{ flexDirection: "column", gap: "1rem", padding: "4rem" }}>
          <div className="spinner" style={{ width: "40px", height: "40px" }} />
          <p>Running model evaluation on all seed reviews...</p>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>This may take a moment on first run as the model loads</p>
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {results && !loading && (
        <>
          <div className="stats-grid" style={{ gridTemplateColumns: "repeat(5, 1fr)" }}>
            <div className="stat-card">
              <div className="stat-value">{(results.stats.accuracy * 100).toFixed(1)}%</div>
              <div className="stat-label">Accuracy</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{(results.stats.macroF1 * 100).toFixed(1)}%</div>
              <div className="stat-label">Macro F1</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{results.stats.correct}/{results.stats.total}</div>
              <div className="stat-label">Correct</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{(results.stats.avgConfidence * 100).toFixed(1)}%</div>
              <div className="stat-label">Avg Confidence</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">3</div>
              <div className="stat-label">Classes</div>
            </div>
          </div>

          <div className="tabs">
            {["overview", "confusion", "per-class", "details"].map(tab => (
              <button key={tab} className={`tab ${activeTab === tab ? "active" : ""}`} onClick={() => setActiveTab(tab)}>
                {tab === "overview" ? "Overview" : tab === "confusion" ? "Confusion Matrix" : tab === "per-class" ? "Per-Class Metrics" : "Review Details"}
              </button>
            ))}
          </div>

          {activeTab === "overview" && (
            <div className="grid-2">
              <div className="card">
                <h3 className="card-title">Sentiment Distribution</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie data={Object.entries(results.stats.sentimentDist).map(([name, value]) => ({ name, value }))} cx="50%" cy="50%" innerRadius={60} outerRadius={100} paddingAngle={4} dataKey="value" label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                      {Object.entries(results.stats.sentimentDist).map(([name]) => (
                        <Cell key={name} fill={COLORS[name]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="card">
                <h3 className="card-title">Expected vs Predicted</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={["Positive", "Neutral", "Negative"].map(s => ({ sentiment: s, Expected: results.stats.sentimentDist[s], Predicted: results.stats.predDist[s] }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="sentiment" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="Expected" fill="#6366f1" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Predicted" fill="#a78bfa" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="card">
                <h3 className="card-title">Confidence Distribution</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={results.stats.confidences.map(c => ({ review: `#${c.id}`, confidence: (c.confidence * 100).toFixed(1), fill: c.correct ? "#10b981" : "#ef4444" }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="review" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Bar dataKey="confidence" radius={[4, 4, 0, 0]}>
                      {results.stats.confidences.map((c, i) => (
                        <Cell key={i} fill={c.correct ? "#10b981" : "#ef4444"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="card">
                <h3 className="card-title">Per-Class Performance Radar</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <RadarChart data={Object.entries(results.stats.perClass).map(([cls, m]) => ({ metric: cls, Precision: +(m.precision * 100).toFixed(1), Recall: +(m.recall * 100).toFixed(1), F1: +(m.f1 * 100).toFixed(1) }))}>
                    <PolarGrid stroke="#e2e8f0" />
                    <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
                    <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                    <Radar name="Precision" dataKey="Precision" stroke="#6366f1" fill="#6366f1" fillOpacity={0.15} />
                    <Radar name="Recall" dataKey="Recall" stroke="#10b981" fill="#10b981" fillOpacity={0.15} />
                    <Radar name="F1" dataKey="F1" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.15} />
                    <Legend />
                    <Tooltip />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {activeTab === "confusion" && (
            <div className="card">
              <h3 className="card-title">Confusion Matrix</h3>
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Actual \ Predicted</th>
                      <th style={{ color: COLORS.Positive }}>Positive</th>
                      <th style={{ color: COLORS.Neutral }}>Neutral</th>
                      <th style={{ color: COLORS.Negative }}>Negative</th>
                      <th>Total</th>
                    </tr>
                  </thead>
                  <tbody>
                    {["Positive", "Neutral", "Negative"].map(actual => {
                      const row = results.stats.confusionMatrix[actual];
                      const total = Object.values(row).reduce((s, v) => s + v, 0);
                      return (
                        <tr key={actual}>
                          <td><strong style={{ color: COLORS[actual] }}>{actual}</strong></td>
                          {["Positive", "Neutral", "Negative"].map(pred => (
                            <td key={pred} style={{
                              background: actual === pred ? "rgba(99, 102, 241, 0.08)" : row[pred] > 0 ? "rgba(239, 68, 68, 0.05)" : "transparent",
                              fontWeight: actual === pred ? "700" : "400",
                              fontSize: "1.1rem",
                              textAlign: "center"
                            }}>
                              {row[pred]}
                            </td>
                          ))}
                          <td style={{ textAlign: "center", fontWeight: "600" }}>{total}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p style={{ marginTop: "1rem", fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                Diagonal values (highlighted) represent correct predictions. Off-diagonal values indicate misclassifications.
              </p>
            </div>
          )}

          {activeTab === "per-class" && (
            <div>
              <div className="grid-3">
                {Object.entries(results.stats.perClass).map(([cls, metrics]) => (
                  <div className="card card-accent" key={cls}>
                    <h3 className="card-title">
                      <span className={`sentiment-badge sentiment-${cls}`}>{cls}</span>
                    </h3>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
                      <div>
                        <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: "0.25rem" }}>Precision</div>
                        <div style={{ fontSize: "1.5rem", fontWeight: "700" }}>{(metrics.precision * 100).toFixed(1)}%</div>
                      </div>
                      <div>
                        <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: "0.25rem" }}>Recall</div>
                        <div style={{ fontSize: "1.5rem", fontWeight: "700" }}>{(metrics.recall * 100).toFixed(1)}%</div>
                      </div>
                      <div>
                        <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: "0.25rem" }}>F1 Score</div>
                        <div style={{ fontSize: "1.5rem", fontWeight: "700", color: "var(--primary)" }}>{(metrics.f1 * 100).toFixed(1)}%</div>
                      </div>
                      <div>
                        <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: "0.25rem" }}>Support</div>
                        <div style={{ fontSize: "1.5rem", fontWeight: "700" }}>{metrics.tp + metrics.fn}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="card" style={{ marginTop: "1.5rem" }}>
                <h3 className="card-title">Classification Report</h3>
                <div className="table-container">
                  <table>
                    <thead>
                      <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(results.stats.perClass).map(([cls, m]) => (
                        <tr key={cls}>
                          <td><span className={`sentiment-badge sentiment-${cls}`}>{cls}</span></td>
                          <td>{(m.precision * 100).toFixed(1)}%</td>
                          <td>{(m.recall * 100).toFixed(1)}%</td>
                          <td style={{ fontWeight: "600" }}>{(m.f1 * 100).toFixed(1)}%</td>
                          <td>{m.tp + m.fn}</td>
                        </tr>
                      ))}
                      <tr style={{ borderTop: "2px solid var(--border)", fontWeight: "700" }}>
                        <td>Macro Avg</td>
                        <td>{(Object.values(results.stats.perClass).reduce((s, m) => s + m.precision, 0) / 3 * 100).toFixed(1)}%</td>
                        <td>{(Object.values(results.stats.perClass).reduce((s, m) => s + m.recall, 0) / 3 * 100).toFixed(1)}%</td>
                        <td style={{ color: "var(--primary)" }}>{(results.stats.macroF1 * 100).toFixed(1)}%</td>
                        <td>{results.stats.total}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {activeTab === "details" && (
            <div className="card">
              <h3 className="card-title">Individual Review Predictions</h3>
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Review</th>
                      <th>Expected</th>
                      <th>Predicted</th>
                      <th>Confidence</th>
                      <th>Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.reviews.map(r => {
                      const pred = r.prediction?.document_level?.overall_sentiment;
                      const conf = r.prediction?.document_level?.confidence || 0;
                      const isCorrect = r.expected_sentiment === pred;
                      return (
                        <tr key={r.id}>
                          <td>{r.id}</td>
                          <td style={{ maxWidth: "350px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: "0.8rem" }}>{r.text}</td>
                          <td><span className={`sentiment-badge sentiment-${r.expected_sentiment}`}>{r.expected_sentiment}</span></td>
                          <td><span className={`sentiment-badge sentiment-${pred}`}>{pred}</span></td>
                          <td style={{ fontVariantNumeric: "tabular-nums" }}>{(conf * 100).toFixed(1)}%</td>
                          <td style={{ fontSize: "1.1rem" }}>{isCorrect ? "✓" : "✗"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div style={{ textAlign: "center", marginTop: "1rem" }}>
            <button className="btn btn-secondary" onClick={runFullEvaluation}>Re-run Evaluation</button>
          </div>
        </>
      )}
    </div>
  );
}

export default Results;
