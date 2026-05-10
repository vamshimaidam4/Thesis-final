import axios from "axios";

// Auto-detect: use current origin in production, localhost in dev
const API_BASE = import.meta.env.VITE_API_URL || window.location.origin;

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
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

// Training endpoints
export const startTraining = (config) =>
  api.post("/api/training/start", config).then((r) => r.data);

export const getTrainingStatus = () =>
  api.get("/api/training/status").then((r) => r.data);

export const getTrainingHistory = () =>
  api.get("/api/training/history").then((r) => r.data);

export const getTrainedModels = () =>
  api.get("/api/training/models").then((r) => r.data);

// SageMaker (cloud training) endpoints
export const getSageMakerStatus = () =>
  api.get("/api/sagemaker/status").then((r) => r.data);

export const startSageMakerTraining = (config) =>
  api.post("/api/sagemaker/train", config).then((r) => r.data);

export const listSageMakerJobs = (limit = 20) =>
  api.get(`/api/sagemaker/jobs?limit=${limit}`).then((r) => r.data);

export const describeSageMakerJob = (name) =>
  api.get(`/api/sagemaker/jobs/${encodeURIComponent(name)}`).then((r) => r.data);

export const syncSageMakerArtifacts = (name) =>
  api.post(`/api/sagemaker/jobs/${encodeURIComponent(name)}/sync`).then((r) => r.data);

export const stopSageMakerJob = (name) =>
  api.post(`/api/sagemaker/jobs/${encodeURIComponent(name)}/stop`).then((r) => r.data);

export const reloadInferenceModels = () =>
  api.post("/api/sagemaker/reload").then((r) => r.data);

export default api;
