import { useState } from "react";

const MODES = {
  ask: {
    label: "Ask",
    url: "/api/ask",
    field: "question",
    placeholder: "Ask a question...",
    resultLabel: "Answer",
  },
  summarize: {
    label: "Summarize",
    url: "/api/summarize",
    field: "text",
    placeholder: "Enter text to summarize...",
    resultLabel: "Summary",
  },
  sentiment: {
    label: "Analyze Sentiment",
    url: "/api/analyze-sentiment",
    field: "text",
    placeholder: "Enter text to analyze...",
    resultLabel: "Sentiment",
  },
};

function formatResult(mode, data) {
  if (mode === "ask") return data.answer.text;
  if (mode === "summarize") return data.summary.text;
  if (mode === "sentiment") {
    return `${data.sentiment.label} (confidence: ${data.sentiment.confidence.toFixed(2)})`;
  }
  return "";
}

function App() {
  const [mode, setMode] = useState("ask");
  const [inputText, setInputText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState("");
  const [ttft, setTtft] = useState(null);
  const [responseTime, setResponseTime] = useState(null);
  const [error, setError] = useState(null);

  const handleModeChange = (newMode) => {
    setMode(newMode);
    setResult("");
    setTtft(null);
    setResponseTime(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!inputText.trim() || loading) return;

    const { url, field } = MODES[mode];

    setLoading(true);
    setError(null);
    setResult("");
    setTtft(null);
    setResponseTime(null);

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [field]: inputText }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.detail || `Request failed (HTTP ${res.status})`);
        return;
      }

      setResult(formatResult(mode, data));
      setTtft(data.ttft_seconds);
      setResponseTime(data.response_time_seconds);
    } catch (err) {
      setError("Could not reach the backend. Is it running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>reportagent</h1>

      <div className="mode-selector">
        {Object.entries(MODES).map(([value, { label }]) => (
          <label
            key={value}
            className={`mode-option ${mode === value ? "active" : ""}`}
          >
            <input
              type="radio"
              name="mode"
              value={value}
              checked={mode === value}
              onChange={() => handleModeChange(value)}
            />
            {label}
          </label>
        ))}
      </div>

      <div className="ask-bar">
        <input
          type="text"
          placeholder={MODES[mode].placeholder}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
        />
        <button type="button" onClick={handleSubmit} disabled={loading}>
          {loading ? "Waiting for the assistant..." : "Submit"}
        </button>
      </div>

      {error && <div className="error-box">{error}</div>}

      <div className="result">
        <label htmlFor="result-box">{MODES[mode].resultLabel}</label>
        <textarea id="result-box" readOnly rows={6} value={result} />

        <div className="metrics">
          <div className="metric-box">
            <label htmlFor="ttft-box">TTFT (s)</label>
            <input
              id="ttft-box"
              type="text"
              readOnly
              value={ttft !== null ? ttft.toFixed(3) : ""}
            />
          </div>
          <div className="metric-box">
            <label htmlFor="total-time-box">Total response time (s)</label>
            <input
              id="total-time-box"
              type="text"
              readOnly
              value={responseTime !== null ? responseTime.toFixed(3) : ""}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
