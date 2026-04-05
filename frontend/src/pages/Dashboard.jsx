import { useState, useEffect } from "react";
import { getHealth, getAWSStatus, getSeedStats } from "../services/api";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import { Activity, Cloud, Database, Brain, Cpu, Server, Shield, Zap } from "lucide-react";

const COLORS = { Positive: "#10b981", Neutral: "#f59e0b", Negative: "#ef4444" };

function Dashboard() {
  const [health, setHealth] = useState(null);
  const [aws, setAws] = useState(null);
  const [seedStats, setSeedStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.allSettled([
      getHealth().catch(() => null),
      getAWSStatus().catch(() => ({ status: "unavailable" })),
      getSeedStats().catch(() => null),
    ]).then(([h, a, s]) => {
      setHealth(h.value);
      setAws(a.value);
      setSeedStats(s.value);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner" />
        Loading dashboard...
      </div>
    );
  }

  const isOnline = health?.status === "healthy";

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">System Dashboard</h2>
        <p className="page-description">Overview of the multi-level sentiment analysis platform</p>
      </div>

      <div className="stats-grid" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
        <div className="stat-card">
          <div className="stat-value">3</div>
          <div className="stat-label">Analysis Levels</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">4</div>
          <div className="stat-label">Model Variants</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{seedStats?.count || 0}</div>
          <div className="stat-label">Seed Reviews</div>
        </div>
        <div className="stat-card">
          <div className="stat-value" style={{ fontSize: "1.4rem" }}>
            <span className={`status-dot ${isOnline ? "online" : "offline"}`} />
            {isOnline ? "Online" : "Offline"}
          </div>
          <div className="stat-label">API Status</div>
        </div>
      </div>

      <div className="grid-2">
        <div className="card card-accent">
          <h3 className="card-title"><Cpu size={18} /> System Information</h3>
          {health ? (
            <div style={{ display: "grid", gap: "0.75rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem 0", borderBottom: "1px solid var(--border)" }}>
                <span style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>Status</span>
                <span className="success" style={{ padding: "0.125rem 0.75rem", fontSize: "0.8rem" }}>{health.status}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem 0", borderBottom: "1px solid var(--border)" }}>
                <span style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>Base Model</span>
                <span style={{ fontWeight: "600", fontSize: "0.875rem" }}>{health.model}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem 0", borderBottom: "1px solid var(--border)" }}>
                <span style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>Compute Device</span>
                <span style={{ fontWeight: "600", fontSize: "0.875rem" }}>{health.device?.device?.toUpperCase()}</span>
              </div>
              {health.device?.gpu && (
                <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem 0" }}>
                  <span style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>GPU</span>
                  <span style={{ fontWeight: "600", fontSize: "0.875rem" }}>{health.device.gpu}</span>
                </div>
              )}
            </div>
          ) : (
            <div className="error">
              Backend not reachable. Start the server:<br />
              <code style={{ fontSize: "0.8rem" }}>cd backend && uvicorn app.main:app --reload</code>
            </div>
          )}
        </div>

        <div className="card card-accent">
          <h3 className="card-title"><Cloud size={18} /> AWS Cloud Status</h3>
          {aws ? (
            <div style={{ display: "grid", gap: "0.75rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem 0", borderBottom: "1px solid var(--border)" }}>
                <span style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>Connection</span>
                <span className={aws.status === "connected" ? "success" : "error"} style={{ padding: "0.125rem 0.75rem", fontSize: "0.8rem" }}>
                  {aws.status}
                </span>
              </div>
              {aws.region && (
                <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem 0", borderBottom: "1px solid var(--border)" }}>
                  <span style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>Region</span>
                  <span style={{ fontWeight: "600", fontSize: "0.875rem" }}>{aws.region}</span>
                </div>
              )}
              {aws.bucket && (
                <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem 0" }}>
                  <span style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>S3 Bucket</span>
                  <span style={{ fontWeight: "600", fontSize: "0.875rem", fontFamily: "monospace" }}>{aws.bucket}</span>
                </div>
              )}
              {aws.error && <div className="error" style={{ marginTop: "0.5rem" }}>{aws.error}</div>}
            </div>
          ) : (
            <p style={{ color: "var(--text-secondary)" }}>AWS credentials not configured</p>
          )}
        </div>
      </div>

      <div className="card">
        <h3 className="card-title"><Brain size={18} /> Available Models</h3>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Architecture</th>
                <th>Analysis Level</th>
                <th>Key Features</th>
                <th>Expected Performance</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong style={{ color: "var(--primary)" }}>HMGS</strong></td>
                <td>BERT + Multi-Head Attention + CRF</td>
                <td>Document, Sentence, Aspect</td>
                <td>Multi-task learning, hierarchical aggregation</td>
                <td><span className="sentiment-badge sentiment-Positive">~93% Acc</span></td>
              </tr>
              <tr>
                <td><strong style={{ color: "var(--primary)" }}>BERT-BiLSTM</strong></td>
                <td>BERT + BiLSTM + Attention</td>
                <td>Document</td>
                <td>Sequential modeling, token attention</td>
                <td><span className="sentiment-badge sentiment-Positive">~90% Acc</span></td>
              </tr>
              <tr>
                <td><strong>BERT-Linear</strong></td>
                <td>BERT + Linear Head</td>
                <td>Document</td>
                <td>Baseline, [CLS] classification</td>
                <td><span className="sentiment-badge sentiment-Neutral">~87% Acc</span></td>
              </tr>
              <tr>
                <td><strong>TF-IDF + LR</strong></td>
                <td>TF-IDF + Logistic Regression</td>
                <td>Document</td>
                <td>Traditional ML baseline</td>
                <td><span className="sentiment-badge sentiment-Neutral">~78% Acc</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {seedStats && seedStats.sentiments && (
        <div className="grid-2">
          <div className="card">
            <h3 className="card-title"><Database size={18} /> Seed Data Distribution</h3>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={Object.entries(seedStats.sentiments).map(([name, value]) => ({ name, value }))}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={90}
                  paddingAngle={4}
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {Object.entries(seedStats.sentiments).map(([name]) => (
                    <Cell key={name} fill={COLORS[name]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h3 className="card-title"><Activity size={18} /> Quick Stats</h3>
            <div style={{ display: "grid", gap: "1rem", padding: "1rem 0" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.75rem", background: "#f8fafc", borderRadius: "var(--radius-sm)" }}>
                <span style={{ color: "var(--text-secondary)" }}>Total Reviews</span>
                <span style={{ fontSize: "1.25rem", fontWeight: "700" }}>{seedStats.count}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.75rem", background: "#f8fafc", borderRadius: "var(--radius-sm)" }}>
                <span style={{ color: "var(--text-secondary)" }}>Avg Review Length</span>
                <span style={{ fontSize: "1.25rem", fontWeight: "700" }}>{seedStats.avg_length?.toFixed(0)} words</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.75rem", background: "#f8fafc", borderRadius: "var(--radius-sm)" }}>
                <span style={{ color: "var(--text-secondary)" }}>Categories</span>
                <span style={{ fontSize: "1.25rem", fontWeight: "700" }}>{Object.keys(seedStats.categories || {}).length}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.75rem", background: "#f8fafc", borderRadius: "var(--radius-sm)" }}>
                <span style={{ color: "var(--text-secondary)" }}>Sentiment Classes</span>
                <span style={{ fontSize: "1.25rem", fontWeight: "700" }}>{Object.keys(seedStats.sentiments).length}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <h3 className="card-title"><Server size={18} /> AWS Infrastructure</h3>
        <div className="grid-4">
          {[
            { icon: <Database size={20} />, service: "S3", desc: "Model checkpoint storage & versioning" },
            { icon: <Cpu size={20} />, service: "EC2 / SageMaker", desc: "GPU training (p3.2xlarge)" },
            { icon: <Shield size={20} />, service: "IAM", desc: "Access control & permissions" },
            { icon: <Zap size={20} />, service: "CloudWatch", desc: "Metrics monitoring & logging" },
          ].map((item, i) => (
            <div key={i} style={{ padding: "1.25rem", background: "#f8fafc", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", textAlign: "center" }}>
              <div style={{ color: "var(--primary)", marginBottom: "0.5rem" }}>{item.icon}</div>
              <div style={{ fontWeight: "700", fontSize: "0.875rem", marginBottom: "0.25rem" }}>{item.service}</div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>{item.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
