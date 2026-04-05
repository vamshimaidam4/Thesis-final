import { useState } from "react";
import { analyzeBatch } from "../services/api";
import { BarChart3, Upload, Download } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";

const COLORS = { Positive: "#10b981", Neutral: "#f59e0b", Negative: "#ef4444" };

function BatchAnalysis() {
  const [text, setText] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleBatch = async () => {
    const reviews = text.split("\n").map((r) => r.trim()).filter(Boolean);
    if (reviews.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeBatch(reviews);
      setResults(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
    setLoading(false);
  };

  const loadSamples = () => {
    setText(
      [
        "Amazing product, works perfectly and arrived on time! Battery life is incredible.",
        "Terrible quality, broke after one week of use. The screen cracked easily.",
        "It's okay, nothing special but does the job. Average build quality.",
        "The best headphones I've ever owned. Sound quality is incredible and comfortable to wear.",
        "Waste of money. The keyboard stopped working after a month. Very disappointed.",
        "Decent camera for the price. Not the best but takes acceptable photos.",
        "Absolutely love this tablet! Fast, responsive, and beautiful display.",
        "Poor customer service and the product arrived damaged. Would not recommend.",
      ].join("\n")
    );
  };

  const reviewCount = text.split("\n").filter(Boolean).length;

  const exportCSV = () => {
    if (!results) return;
    const rows = [["#", "Review", "Sentiment", "Confidence", "Aspects"]];
    results.results.forEach((r, i) => {
      const aspects = r.aspect_level?.aspects?.map(a => `${a.aspect}(${a.sentiment})`).join("; ") || "";
      rows.push([i + 1, `"${r.text}"`, r.document_level?.overall_sentiment, ((r.document_level?.confidence || 0) * 100).toFixed(1) + "%", aspects]);
    });
    const csv = rows.map(r => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "sentiment_analysis_results.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Batch Analysis</h2>
        <p className="page-description">Analyze multiple reviews simultaneously with comprehensive results</p>
      </div>

      <div className="card card-accent">
        <h3 className="card-title"><Upload size={18} /> Enter Reviews (one per line, max 50)</h3>
        <div className="textarea-wrapper">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter one review per line for batch processing..."
            style={{ minHeight: "200px" }}
          />
        </div>
        <div className="btn-group">
          <button className="btn btn-primary" onClick={handleBatch} disabled={loading || !text.trim()}>
            <BarChart3 size={16} />
            {loading ? "Analyzing..." : `Analyze ${reviewCount} Review${reviewCount !== 1 ? "s" : ""}`}
          </button>
          <button className="btn btn-secondary" onClick={loadSamples}>
            Load Samples
          </button>
          {text && <button className="btn btn-secondary btn-sm" onClick={() => { setText(""); setResults(null); }}>Clear</button>}
        </div>
      </div>

      {loading && (
        <div className="loading" style={{ flexDirection: "column", gap: "0.75rem" }}>
          <div className="spinner" />
          <span>Processing {reviewCount} reviews...</span>
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {results && !loading && (
        <>
          <div className="grid-2" style={{ marginBottom: "1.5rem" }}>
            <div className="card">
              <h3 className="card-title">Sentiment Distribution</h3>
              <div className="stats-grid" style={{ gridTemplateColumns: "repeat(3, 1fr)", marginBottom: 0 }}>
                {["Positive", "Neutral", "Negative"].map((s) => {
                  const count = results.results.filter(r => r.document_level?.overall_sentiment === s).length;
                  const pct = results.count > 0 ? (count / results.count * 100).toFixed(0) : 0;
                  return (
                    <div key={s} style={{ textAlign: "center" }}>
                      <div style={{ fontSize: "2rem", fontWeight: "800", color: COLORS[s] }}>{count}</div>
                      <span className={`sentiment-badge sentiment-${s}`}>{s}</span>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>{pct}%</div>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="card">
              <h3 className="card-title">Distribution Chart</h3>
              <ResponsiveContainer width="100%" height={160}>
                <PieChart>
                  <Pie
                    data={["Positive", "Neutral", "Negative"].map(s => ({ name: s, value: results.results.filter(r => r.document_level?.overall_sentiment === s).length })).filter(d => d.value > 0)}
                    cx="50%" cy="50%" innerRadius={40} outerRadius={65} paddingAngle={4} dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {["Positive", "Neutral", "Negative"].map(s => <Cell key={s} fill={COLORS[s]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
              <h3 className="card-title" style={{ marginBottom: 0 }}>Results ({results.count} reviews)</h3>
              <button className="btn btn-secondary btn-sm" onClick={exportCSV}>
                <Download size={14} /> Export CSV
              </button>
            </div>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Review</th>
                    <th>Sentiment</th>
                    <th>Confidence</th>
                    <th>Aspects</th>
                  </tr>
                </thead>
                <tbody>
                  {results.results.map((r, i) => (
                    <tr key={i}>
                      <td style={{ fontWeight: "600", color: "var(--text-muted)" }}>{i + 1}</td>
                      <td style={{ maxWidth: "350px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: "0.8rem" }}>
                        {r.text}
                      </td>
                      <td>
                        <span className={`sentiment-badge sentiment-${r.document_level?.overall_sentiment}`}>
                          {r.document_level?.overall_sentiment}
                        </span>
                      </td>
                      <td style={{ fontVariantNumeric: "tabular-nums", fontWeight: "600" }}>
                        {((r.document_level?.confidence || 0) * 100).toFixed(1)}%
                      </td>
                      <td>
                        {r.aspect_level?.aspects?.map((a, j) => (
                          <span key={j} className={`sentiment-badge sentiment-${a.sentiment}`} style={{ margin: "0 2px", fontSize: "0.7rem", padding: "0.1rem 0.5rem" }}>
                            {a.aspect}
                          </span>
                        ))}
                        {(!r.aspect_level?.aspects || r.aspect_level.aspects.length === 0) && <span style={{ color: "var(--text-muted)" }}>--</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default BatchAnalysis;
