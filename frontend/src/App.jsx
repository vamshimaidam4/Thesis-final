import { BrowserRouter as Router, Routes, Route, NavLink } from "react-router-dom";
import { Brain, BarChart3, Database, Layers, FlaskConical, LayoutDashboard } from "lucide-react";
import Dashboard from "./pages/Dashboard";
import Analyzer from "./pages/Analyzer";
import BatchAnalysis from "./pages/BatchAnalysis";
import SeedData from "./pages/SeedData";
import Architecture from "./pages/Architecture";
import Results from "./pages/Results";
import "./App.css";

function App() {
  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <div className="nav-brand">
            <Brain size={28} className="brand-icon" />
            <div>
              <h1>Sentiment Analysis Platform</h1>
              <span className="nav-subtitle">Multi-Level Deep Learning for E-Commerce Reviews</span>
            </div>
          </div>
          <div className="nav-links">
            <NavLink to="/" end><LayoutDashboard size={16} /> Dashboard</NavLink>
            <NavLink to="/analyze"><FlaskConical size={16} /> Analyze</NavLink>
            <NavLink to="/batch"><BarChart3 size={16} /> Batch</NavLink>
            <NavLink to="/results"><Layers size={16} /> Results</NavLink>
            <NavLink to="/seed-data"><Database size={16} /> Seed Data</NavLink>
            <NavLink to="/architecture"><Brain size={16} /> Architecture</NavLink>
          </div>
        </nav>
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/analyze" element={<Analyzer />} />
            <Route path="/batch" element={<BatchAnalysis />} />
            <Route path="/results" element={<Results />} />
            <Route path="/seed-data" element={<SeedData />} />
            <Route path="/architecture" element={<Architecture />} />
          </Routes>
        </main>
        <footer className="footer">
          <p>Hierarchical Multi-Granularity Sentiment Analysis &middot; BERT-BiLSTM-Attention &middot; Masters Thesis Research</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;
