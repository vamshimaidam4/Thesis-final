import { useState, useEffect, useRef } from "react";
import { startTraining, getTrainingStatus, getTrainingHistory, getTrainedModels } from "../services/api";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Play, Square, RefreshCw, Download, Cpu, CheckCircle, XCircle, Clock } from "lucide-react";
import SageMakerPanel from "../components/SageMakerPanel";

function Training() {
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState(null);
  const [models, setModels] = useState([]);
  const [config, setConfig] = useState({ model: "all", epochs: 3, batch_size: 8, train_samples: 600 });
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => {
    loadData();
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const loadData = async () => {
    try {
      const [s, h, m] = await Promise.all([
        getTrainingStatus().catch(() => null),
        getTrainingHistory().catch(() => null),
        getTrainedModels().catch(() => ({ models: [] })),
      ]);
      setStatus(s);
      setHistory(h);
      setModels(m.models || []);
      if (s && s.status === "training") startPolling();
    } catch {}
  };

  const startPolling = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await getTrainingStatus();
        setStatus(s);
        if (s.status !== "training" && s.status !== "starting") {
          clearInterval(pollRef.current);
          pollRef.current = null;
          const [h, m] = await Promise.all([getTrainingHistory(), getTrainedModels()]);
          setHistory(h);
          setModels(m.models || []);
        }
      } catch {}
    }, 3000);
  };

  const handleStart = async () => {
    setError(null);
    try {
      await startTraining(config);
      startPolling();
      const s = await getTrainingStatus();
      setStatus(s);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
  };

  const isTraining = status?.status === "training" || status?.status === "starting";

  const chartData = status?.history?.length > 0 ? status.history : [];

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Model Training</h2>
        <p className="page-description">Train sentiment analysis models directly from the browser</p>
      </div>

      {/* Cloud (SageMaker) training */}
      <SageMakerPanel />

      {/* Training Control Panel */}
      <div className="card card-accent">
        <h3 className="card-title"><Cpu size={18} /> Training Configuration</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem", marginBottom: "1.5rem" }}>
          <div>
            <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", display: "block", marginBottom: "0.375rem", fontWeight: "600" }}>Model</label>
            <select value={config.model} onChange={e => setConfig({ ...config, model: e.target.value })} disabled={isTraining}
              style={{ width: "100%", padding: "0.5rem", border: "2px solid var(--border)", borderRadius: "8px", fontSize: "0.875rem", fontFamily: "inherit" }}>
              <option value="all">All Models</option>
              <option value="hmgs">HMGS Only</option>
              <option value="bilstm">BiLSTM Only</option>
            </select>
          </div>
          <div>
            <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", display: "block", marginBottom: "0.375rem", fontWeight: "600" }}>Epochs</label>
            <input type="number" min="1" max="20" value={config.epochs} onChange={e => setConfig({ ...config, epochs: parseInt(e.target.value) || 1 })} disabled={isTraining}
              style={{ width: "100%", padding: "0.5rem", border: "2px solid var(--border)", borderRadius: "8px", fontSize: "0.875rem", fontFamily: "inherit" }} />
          </div>
          <div>
            <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", display: "block", marginBottom: "0.375rem", fontWeight: "600" }}>Batch Size</label>
            <input type="number" min="2" max="64" value={config.batch_size} onChange={e => setConfig({ ...config, batch_size: parseInt(e.target.value) || 8 })} disabled={isTraining}
              style={{ width: "100%", padding: "0.5rem", border: "2px solid var(--border)", borderRadius: "8px", fontSize: "0.875rem", fontFamily: "inherit" }} />
          </div>
          <div>
            <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", display: "block", marginBottom: "0.375rem", fontWeight: "600" }}>Training Samples</label>
            <input type="number" min="100" max="50000" step="100" value={config.train_samples} onChange={e => setConfig({ ...config, train_samples: parseInt(e.target.value) || 600 })} disabled={isTraining}
              style={{ width: "100%", padding: "0.5rem", border: "2px solid var(--border)", borderRadius: "8px", fontSize: "0.875rem", fontFamily: "inherit" }} />
          </div>
        </div>
        <div className="btn-group">
          <button className="btn btn-primary" onClick={handleStart} disabled={isTraining} style={{ padding: "0.75rem 2rem" }}>
            <Play size={16} /> {isTraining ? "Training in Progress..." : "Start Training"}
          </button>
          <button className="btn btn-secondary" onClick={loadData}>
            <RefreshCw size={16} /> Refresh Status
          </button>
        </div>
        {error && <div className="error" style={{ marginTop: "1rem" }}>{error}</div>}
      </div>

      {/* Training Progress */}
      {status && status.status !== "idle" && (
        <div className="card">
          <h3 className="card-title">
            {status.status === "training" || status.status === "starting" ? <><div className="spinner" style={{ width: "18px", height: "18px", margin: 0 }} /> Training Progress</> :
             status.status === "completed" ? <><CheckCircle size={18} color="var(--positive)" /> Training Complete</> :
             status.status === "failed" ? <><XCircle size={18} color="var(--negative)" /> Training Failed</> :
             <><Clock size={18} /> Training Status</>}
          </h3>

          {/* Progress bar */}
          <div style={{ marginBottom: "1.5rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.375rem" }}>
              <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>{status.message}</span>
              <span style={{ fontSize: "0.8rem", fontWeight: "700" }}>{status.progress}%</span>
            </div>
            <div style={{ height: "12px", background: "#e2e8f0", borderRadius: "6px", overflow: "hidden" }}>
              <div style={{
                height: "100%",
                width: `${status.progress}%`,
                background: status.status === "failed" ? "var(--negative)" : "linear-gradient(90deg, var(--primary), #8b5cf6)",
                borderRadius: "6px",
                transition: "width 0.5s"
              }} />
            </div>
          </div>

          {/* Live metrics */}
          <div className="stats-grid" style={{ gridTemplateColumns: "repeat(5, 1fr)" }}>
            <div className="stat-card" style={{ padding: "1rem" }}>
              <div className="stat-value" style={{ fontSize: "1.5rem" }}>{status.model?.toUpperCase() || "-"}</div>
              <div className="stat-label">Current Model</div>
            </div>
            <div className="stat-card" style={{ padding: "1rem" }}>
              <div className="stat-value" style={{ fontSize: "1.5rem" }}>{status.epoch}/{status.total_epochs}</div>
              <div className="stat-label">Epoch</div>
            </div>
            <div className="stat-card" style={{ padding: "1rem" }}>
              <div className="stat-value" style={{ fontSize: "1.5rem" }}>{status.train_loss ? status.train_loss.toFixed(3) : "-"}</div>
              <div className="stat-label">Train Loss</div>
            </div>
            <div className="stat-card" style={{ padding: "1rem" }}>
              <div className="stat-value" style={{ fontSize: "1.5rem" }}>{status.val_f1 ? (status.val_f1 * 100).toFixed(1) + "%" : "-"}</div>
              <div className="stat-label">Val F1</div>
            </div>
            <div className="stat-card" style={{ padding: "1rem" }}>
              <div className="stat-value" style={{ fontSize: "1.5rem" }}>{status.val_acc ? (status.val_acc * 100).toFixed(1) + "%" : "-"}</div>
              <div className="stat-label">Val Accuracy</div>
            </div>
          </div>

          {status.elapsed_seconds && (
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.5rem" }}>
              Elapsed: {Math.floor(status.elapsed_seconds / 60)}m {status.elapsed_seconds % 60}s
            </p>
          )}
        </div>
      )}

      {/* Training Curves */}
      {chartData.length > 0 && (
        <div className="grid-2">
          <div className="card">
            <h3 className="card-title">Loss Curves</h3>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="epoch" tick={{ fontSize: 12 }} label={{ value: "Epoch", position: "bottom", fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="train_loss" stroke="#6366f1" strokeWidth={2} name="Train Loss" dot={{ r: 4 }} />
                <Line type="monotone" dataKey="val_loss" stroke="#ef4444" strokeWidth={2} name="Val Loss" dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <h3 className="card-title">Performance Curves</h3>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="epoch" tick={{ fontSize: 12 }} label={{ value: "Epoch", position: "bottom", fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="val_f1" stroke="#10b981" strokeWidth={2} name="Val F1" dot={{ r: 4 }} />
                <Line type="monotone" dataKey="val_acc" stroke="#f59e0b" strokeWidth={2} name="Val Accuracy" dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Trained Models */}
      <div className="card">
        <h3 className="card-title"><Download size={18} /> Trained Model Checkpoints</h3>
        {models.length > 0 ? (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Model File</th>
                  <th>Size</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m, i) => (
                  <tr key={i}>
                    <td style={{ fontFamily: "monospace", fontSize: "0.85rem" }}>{m.name}</td>
                    <td>{m.size_mb} MB</td>
                    <td><span className="sentiment-badge sentiment-Positive">Ready</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p style={{ color: "var(--text-secondary)", padding: "1rem 0" }}>
            No trained models yet. Click "Start Training" above to train your first model.
          </p>
        )}
      </div>

      {/* Saved Training History */}
      {history && Object.keys(history).length > 0 && (
        <div className="card">
          <h3 className="card-title">Saved Training History</h3>
          {Object.entries(history).map(([modelName, epochs]) => (
            <div key={modelName} style={{ marginBottom: "1rem" }}>
              <h4 style={{ fontSize: "0.875rem", fontWeight: "600", marginBottom: "0.5rem", color: "var(--primary)" }}>{modelName.toUpperCase()}</h4>
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Epoch</th>
                      <th>Train Loss</th>
                      <th>Val Loss</th>
                      <th>Val F1</th>
                      <th>Val Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {epochs.map((e, i) => (
                      <tr key={i}>
                        <td>{e.epoch}</td>
                        <td>{e.train_loss?.toFixed(4)}</td>
                        <td>{e.val_loss?.toFixed(4)}</td>
                        <td style={{ fontWeight: "600" }}>{(e.val_f1 * 100).toFixed(1)}%</td>
                        <td>{(e.val_acc * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Training;
