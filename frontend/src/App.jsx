import { useState } from "react";

function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState("");
  const [ttft, setTtft] = useState(null);
  const [responseTime, setResponseTime] = useState(null);
  const [error, setError] = useState(null);

  const handleAsk = async () => {
    if (!question.trim() || loading) return;

    setLoading(true);
    setError(null);
    setAnswer("");
    setTtft(null);
    setResponseTime(null);

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.detail || `Request failed (HTTP ${res.status})`);
        return;
      }

      setAnswer(data.answer.text);
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
      <div className="ask-bar">
        <input
          type="text"
          placeholder="Ask a question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
        />
        <button type="button" onClick={handleAsk} disabled={loading}>
          {loading ? "Waiting for the assistant..." : "Ask"}
        </button>
      </div>

      {error && <div className="error-box">{error}</div>}

      <div className="result">
        <label htmlFor="answer-box">Answer</label>
        <textarea id="answer-box" readOnly rows={6} value={answer} />

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
