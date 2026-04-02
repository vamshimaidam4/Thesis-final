function ProbabilityBars({ probabilities }) {
  if (!probabilities) return null;

  const labels = ["Negative", "Neutral", "Positive"];
  const colors = {
    Negative: "prob-fill-negative",
    Neutral: "prob-fill-neutral",
    Positive: "prob-fill-positive",
  };

  return (
    <div style={{ marginTop: "0.5rem" }}>
      {labels.map((label) => {
        const value = probabilities[label] || 0;
        return (
          <div className="prob-bar-container" key={label}>
            <span className="prob-label">{label}</span>
            <div className="prob-bar">
              <div
                className={`prob-fill ${colors[label]}`}
                style={{ width: `${value * 100}%` }}
              />
            </div>
            <span className="prob-value">{(value * 100).toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}

export default ProbabilityBars;
