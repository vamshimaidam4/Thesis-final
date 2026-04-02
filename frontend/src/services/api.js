import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
});

export const analyzeDocument = (text) =>
  api.post("/api/analysis/document", { text }).then((r) => r.data);

export const analyzeSentence = (text) =>
  api.post("/api/analysis/sentence", { text }).then((r) => r.data);

export const analyzeAspect = (text) =>
  api.post("/api/analysis/aspect", { text }).then((r) => r.data);

export const analyzeFull = (text, model = "hmgs") =>
  api.post("/api/analysis/full", { text, model }).then((r) => r.data);

export const analyzeBatch = (reviews) =>
  api.post("/api/analysis/batch", { reviews }).then((r) => r.data);

export const getHealth = () =>
  api.get("/api/health").then((r) => r.data);

export const getAWSStatus = () =>
  api.get("/api/aws/status").then((r) => r.data);

export const getSeedReviews = () =>
  api.get("/api/seed/reviews").then((r) => r.data);

export const getSeedStats = () =>
  api.get("/api/seed/stats").then((r) => r.data);

export default api;
