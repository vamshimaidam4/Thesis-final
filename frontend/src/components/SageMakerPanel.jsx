import { useState, useEffect, useRef } from "react";
import {
  getSageMakerStatus,
  startSageMakerTraining,
  listSageMakerJobs,
  describeSageMakerJob,
  syncSageMakerArtifacts,
  stopSageMakerJob,
  reloadInferenceModels,
} from "../services/api";
import { Cloud, Play, RefreshCw, ExternalLink, Download, StopCircle, RotateCw } from "lucide-react";

const TERMINAL = new Set(["Completed", "Failed", "Stopped"]);

function fmtTime(iso) {
  if (!iso) return "-";
  return new Date(iso).toLocaleString();
}

function statusColor(s) {
  if (s === "Completed") return "var(--positive)";
  if (s === "Failed") return "var(--negative)";
  if (s === "Stopped") return "var(--text-muted)";
  if (s === "InProgress") return "var(--primary)";
  return "var(--text-secondary)";
}

function SageMakerPanel() {
  const [status, setStatus] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [activeJob, setActiveJob] = useState(null);
  const [config, setConfig] = useState({
    instance_type: "ml.g4dn.xlarge",
    model: "both",
    epochs: 3,
    batch_size: 16,
    train_samples: 10000,
    val_samples: 1000,
    hf_config: "raw_review_All_Beauty",
    max_len: 128,
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [info, setInfo] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => {
    loadStatus();
    loadJobs();
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const loadStatus = async () => {
    try { setStatus(await getSageMakerStatus()); } catch {}
  };

  const loadJobs = async () => {
    try {
      const r = await listSageMakerJobs(20);
      setJobs(r.jobs || []);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    }
  };

  const refreshActive = async (name) => {
    try {
      const d = await describeSageMakerJob(name);
      setActiveJob(d);
      if (TERMINAL.has(d.status) && pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
        loadJobs();
      }
    } catch {}
  };

  const startPolling = (name) => {
    if (pollRef.current) clearInterval(pollRef.current);
    refreshActive(name);
    pollRef.current = setInterval(() => refreshActive(name), 8000);
  };

  const handleSelect = (name) => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    setActiveJob(null);
    setInfo(null);
    setError(null);
    if (name) startPolling(name);
  };

  const handleSubmit = async () => {
    setError(null); setInfo(null); setBusy(true);
    try {
      const r = await startSageMakerTraining(config);
      setInfo(`Submitted ${r.job_name} on ${r.instance_type}.`);
      await loadJobs();
      startPolling(r.job_name);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleSync = async () => {
    if (!activeJob) return;
    setBusy(true); setError(null); setInfo(null);
    try {
      const r = await syncSageMakerArtifacts(activeJob.name);
      setInfo(`Synced ${r.uploaded.length} files to s3://.../${r.inference_prefix}`);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleReload = async () => {
    setBusy(true); setError(null); setInfo(null);
    try {
      await reloadInferenceModels();
      setInfo("Inference cache cleared. Next /api/analysis call will reload weights.");
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleStop = async () => {
    if (!activeJob) return;
    setBusy(true); setError(null);
    try {
      await stopSageMakerJob(activeJob.name);
      setInfo(`Stopping ${activeJob.name}…`);
      refreshActive(activeJob.name);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setBusy(false);
    }
  };

  const roleReady = status?.sagemaker_role_present === true;
  const inputStyle = {
    width: "100%", padding: "0.5rem",
    border: "2px solid var(--border)", borderRadius: "8px",
    fontSize: "0.875rem", fontFamily: "inherit",
  };
  const labelStyle = {
    fontSize: "0.75rem", color: "var(--text-secondary)",
    display: "block", marginBottom: "0.375rem", fontWeight: "600",
  };

  return (
    <div className="card card-accent" style={{ marginBottom: "1.5rem" }}>
      <h3 className="card-title">
        <Cloud size={18} /> Cloud Training (AWS SageMaker)
      </h3>
      <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>
        Trains on real Amazon Reviews 2023 from HuggingFace on a managed SageMaker GPU instance.
        Checkpoints land in <code>s3://{status?.bucket || "…"}/inference/current/</code> and the
        inference backend pulls them automatically on the next request.
      </p>

      {status && (
        <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "1rem", display: "flex", gap: "1.5rem", flexWrap: "wrap" }}>
          <span>account: <strong>{status.account_id || "?"}</strong></span>
          <span>region: <strong>{status.region}</strong></span>
          <span>bucket: <strong>{status.bucket}</strong></span>
          <span>role: <strong style={{ color: roleReady ? "var(--positive)" : "var(--negative)" }}>
            {roleReady ? "ready" : "missing"}
          </strong></span>
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem", marginBottom: "1rem" }}>
        <div>
          <label style={labelStyle}>Instance Type</label>
          <select value={config.instance_type} disabled={busy}
            onChange={e => setConfig({ ...config, instance_type: e.target.value })}
            style={inputStyle}>
            <option value="ml.m5.xlarge">ml.m5.xlarge ($0.23/hr CPU)</option>
            <option value="ml.g4dn.xlarge">ml.g4dn.xlarge ($0.74/hr T4)</option>
            <option value="ml.g5.xlarge">ml.g5.xlarge ($1.41/hr A10G)</option>
            <option value="ml.p3.2xlarge">ml.p3.2xlarge ($3.83/hr V100)</option>
          </select>
        </div>
        <div>
          <label style={labelStyle}>Model</label>
          <select value={config.model} disabled={busy}
            onChange={e => setConfig({ ...config, model: e.target.value })}
            style={inputStyle}>
            <option value="both">Both (HMGS + BiLSTM)</option>
            <option value="hmgs">HMGS only</option>
            <option value="bilstm">BiLSTM only</option>
          </select>
        </div>
        <div>
          <label style={labelStyle}>Epochs</label>
          <input type="number" min={1} max={20} value={config.epochs} disabled={busy}
            onChange={e => setConfig({ ...config, epochs: parseInt(e.target.value) || 1 })}
            style={inputStyle} />
        </div>
        <div>
          <label style={labelStyle}>Train Samples</label>
          <input type="number" min={300} max={200000} step={500} value={config.train_samples} disabled={busy}
            onChange={e => setConfig({ ...config, train_samples: parseInt(e.target.value) || 1000 })}
            style={inputStyle} />
        </div>
        <div>
          <label style={labelStyle}>Val Samples</label>
          <input type="number" min={100} max={20000} step={100} value={config.val_samples} disabled={busy}
            onChange={e => setConfig({ ...config, val_samples: parseInt(e.target.value) || 200 })}
            style={inputStyle} />
        </div>
        <div>
          <label style={labelStyle}>Batch Size</label>
          <input type="number" min={4} max={64} value={config.batch_size} disabled={busy}
            onChange={e => setConfig({ ...config, batch_size: parseInt(e.target.value) || 8 })}
            style={inputStyle} />
        </div>
        <div>
          <label style={labelStyle}>Max Tokens</label>
          <input type="number" min={32} max={512} step={32} value={config.max_len} disabled={busy}
            onChange={e => setConfig({ ...config, max_len: parseInt(e.target.value) || 128 })}
            style={inputStyle} />
        </div>
        <div>
          <label style={labelStyle}>HF Subset</label>
          <input type="text" value={config.hf_config} disabled={busy}
            onChange={e => setConfig({ ...config, hf_config: e.target.value })}
            placeholder="raw_review_All_Beauty"
            style={inputStyle} />
        </div>
      </div>

      <div className="btn-group" style={{ marginBottom: "1rem" }}>
        <button className="btn btn-primary" onClick={handleSubmit}
          disabled={busy || !roleReady} style={{ padding: "0.75rem 1.5rem" }}>
          <Play size={16} /> {busy ? "Submitting…" : "Launch SageMaker Job"}
        </button>
        <button className="btn btn-secondary" onClick={() => { loadStatus(); loadJobs(); }}>
          <RefreshCw size={16} /> Refresh
        </button>
        <button className="btn btn-secondary" onClick={handleReload} disabled={busy}>
          <RotateCw size={16} /> Reload Inference Models
        </button>
      </div>

      {info && <div className="success" style={{ marginBottom: "1rem" }}>{info}</div>}
      {error && <div className="error" style={{ marginBottom: "1rem" }}>{error}</div>}
      {!roleReady && status && (
        <div className="error" style={{ marginBottom: "1rem" }}>
          SageMaker execution role is not present in this account. Re-run the
          "Deploy to AWS" workflow to create it.
        </div>
      )}

      {/* Active job detail */}
      {activeJob && (
        <div className="card" style={{ marginBottom: "1rem", background: "#f8fafc" }}>
          <h4 className="card-title" style={{ fontSize: "0.95rem" }}>
            Active Job: <code>{activeJob.name}</code>
          </h4>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "0.75rem", marginBottom: "0.75rem" }}>
            <div className="stat-card" style={{ padding: "0.75rem" }}>
              <div className="stat-value" style={{ fontSize: "1.1rem", color: statusColor(activeJob.status) }}>
                {activeJob.status}
              </div>
              <div className="stat-label">Status</div>
            </div>
            <div className="stat-card" style={{ padding: "0.75rem" }}>
              <div className="stat-value" style={{ fontSize: "0.9rem" }}>{activeJob.secondary_status || "-"}</div>
              <div className="stat-label">Stage</div>
            </div>
            <div className="stat-card" style={{ padding: "0.75rem" }}>
              <div className="stat-value" style={{ fontSize: "0.9rem" }}>{activeJob.instance_type}</div>
              <div className="stat-label">Instance</div>
            </div>
            <div className="stat-card" style={{ padding: "0.75rem" }}>
              <div className="stat-value" style={{ fontSize: "0.9rem" }}>
                {activeJob.billable_seconds ? Math.round(activeJob.billable_seconds / 60) + " min" : "-"}
              </div>
              <div className="stat-label">Billable</div>
            </div>
          </div>

          {activeJob.transitions?.length > 0 && (
            <div style={{ fontSize: "0.8rem", marginBottom: "0.75rem" }}>
              <strong>Recent transitions:</strong>
              <ul style={{ marginTop: "0.25rem", paddingLeft: "1.25rem" }}>
                {activeJob.transitions.slice(-4).map((t, i) => (
                  <li key={i}>
                    <code>{t.status}</code> — {t.message || ""} ({fmtTime(t.started)})
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="btn-group">
            {activeJob.console_url && (
              <a href={activeJob.console_url} target="_blank" rel="noreferrer" className="btn btn-secondary">
                <ExternalLink size={14} /> AWS Console
              </a>
            )}
            {activeJob.status === "Completed" && (
              <button className="btn btn-primary" onClick={handleSync} disabled={busy}>
                <Download size={14} /> Sync to Inference
              </button>
            )}
            {activeJob.status === "InProgress" && (
              <button className="btn btn-secondary" onClick={handleStop} disabled={busy}>
                <StopCircle size={14} /> Stop Job
              </button>
            )}
          </div>

          {activeJob.model_artifact && (
            <p style={{ fontSize: "0.75rem", marginTop: "0.75rem", color: "var(--text-muted)" }}>
              Artifact: <code>{activeJob.model_artifact}</code>
            </p>
          )}
        </div>
      )}

      {/* Job list */}
      <div>
        <h4 style={{ fontSize: "0.85rem", fontWeight: "600", marginBottom: "0.5rem" }}>
          Recent Jobs ({jobs.length})
        </h4>
        {jobs.length === 0 ? (
          <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
            No SageMaker jobs yet. Launch one above.
          </p>
        ) : (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th>Ended</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {jobs.map(j => (
                  <tr key={j.name}>
                    <td style={{ fontFamily: "monospace", fontSize: "0.8rem" }}>{j.name}</td>
                    <td style={{ color: statusColor(j.status), fontWeight: 600 }}>{j.status}</td>
                    <td style={{ fontSize: "0.8rem" }}>{fmtTime(j.created)}</td>
                    <td style={{ fontSize: "0.8rem" }}>{fmtTime(j.ended)}</td>
                    <td>
                      <button className="btn btn-secondary" style={{ padding: "0.25rem 0.5rem", fontSize: "0.75rem" }}
                        onClick={() => handleSelect(j.name)}>
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default SageMakerPanel;
