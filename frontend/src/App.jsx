import { useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

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

function QueryScreen() {
  const [mode, setMode] = useState("ask");
  const [inputText, setInputText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState("");
  const [sources, setSources] = useState([]);
  const [ttft, setTtft] = useState(null);
  const [responseTime, setResponseTime] = useState(null);
  const [error, setError] = useState(null);

  const handleModeChange = (newMode) => {
    setMode(newMode);
    setResult("");
    setSources([]);
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
    setSources([]);
    setTtft(null);
    setResponseTime(null);

    try {
      const res = await fetch(`${API_BASE_URL}${url}`, {
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
      setSources(mode === "ask" ? data.answer.sources : []);
      setTtft(data.ttft_seconds);
      setResponseTime(data.response_time_seconds);
    } catch (err) {
      setError("Could not reach the backend. Is it running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
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

        {sources.length > 0 && (
          <ol className="sources-list">
            {sources.map((source, i) => (
              <li key={i}>
                {source.type === "web" ? (
                  <>
                    <a href={source.url} target="_blank" rel="noreferrer">
                      {source.title || source.url}
                    </a>
                    {source.snippet && <span className="source-snippet"> — {source.snippet}</span>}
                  </>
                ) : (
                  <>
                    <span className="source-id">
                      {source.document_id} (chunk {source.chunk_id})
                    </span>
                    <span className="source-snippet"> — {source.snippet}</span>
                  </>
                )}
              </li>
            ))}
          </ol>
        )}

        <div className="metrics">
          <div className="metric-box">
            <label htmlFor="ttft-box">TTFT (s)</label>
            <input
              id="ttft-box"
              type="text"
              readOnly
              value={ttft !== null ? ttft.toFixed(1) : ""}
            />
          </div>
          <div className="metric-box">
            <label htmlFor="total-time-box">Total response time (s)</label>
            <input
              id="total-time-box"
              type="text"
              readOnly
              value={responseTime !== null ? responseTime.toFixed(1) : ""}
            />
          </div>
        </div>
      </div>
    </>
  );
}

function IngestScreen() {
  const [documentId, setDocumentId] = useState("");
  const [topic, setTopic] = useState("");
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  const [fileError, setFileError] = useState(null);
  const [extracting, setExtracting] = useState(false);

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-selecting the same file later
    if (!file) return;

    setFileError(null);

    const isPdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
    if (!isPdf) {
      const reader = new FileReader();
      reader.onload = () => setText(reader.result);
      reader.onerror = () => setFileError(`Could not read "${file.name}".`);
      reader.readAsText(file);
      return;
    }

    setExtracting(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_BASE_URL}/api/extract-text`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        setFileError(data.detail || `Could not extract text from "${file.name}".`);
        return;
      }

      setText(data.text);
    } catch (err) {
      setFileError("Could not reach the backend. Is it running?");
    } finally {
      setExtracting(false);
    }
  };

  const handleSubmit = async () => {
    if (!documentId.trim() || !topic.trim() || !text.trim() || loading || extracting) return;

    setLoading(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_id: documentId, topic, text }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.detail || `Request failed (HTTP ${res.status})`);
        return;
      }

      setSuccessMessage(
        `Ingested "${data.document_id}" as ${data.chunks_ingested} chunk${data.chunks_ingested === 1 ? "" : "s"}.`
      );
    } catch (err) {
      setError("Could not reach the backend. Is it running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ingest-form">
      <label htmlFor="ingest-document-id">Document ID</label>
      <input
        id="ingest-document-id"
        type="text"
        placeholder="e.g. policy-handbook-2026"
        value={documentId}
        onChange={(e) => setDocumentId(e.target.value)}
      />

      <label htmlFor="ingest-topic">Topic</label>
      <input
        id="ingest-topic"
        type="text"
        placeholder="e.g. hr-policies"
        value={topic}
        onChange={(e) => setTopic(e.target.value)}
      />

      <label htmlFor="ingest-text">Text</label>
      <textarea
        id="ingest-text"
        rows={10}
        placeholder="Paste the document text here, or upload a .txt or .pdf file below..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <div className="file-upload">
        <label htmlFor="ingest-file">Or upload a .txt or .pdf file:</label>
        <input
          id="ingest-file"
          type="file"
          accept=".txt,text/plain,.pdf,application/pdf"
          onChange={handleFileChange}
          disabled={extracting}
        />
        {extracting && <span className="extracting-label">Extracting text...</span>}
      </div>
      {fileError && <div className="error-box">{fileError}</div>}

      <button type="button" onClick={handleSubmit} disabled={loading || extracting}>
        {loading ? "Ingesting..." : "Ingest"}
      </button>

      {error && <div className="error-box">{error}</div>}
      {successMessage && <div className="success-box">{successMessage}</div>}
    </div>
  );
}

function App() {
  const [screen, setScreen] = useState("ingest");

  return (
    <div className="app">
      <h1>reportagent</h1>

      <div className="screen-selector">
        <button
          type="button"
          className={screen === "ingest" ? "active" : ""}
          onClick={() => setScreen("ingest")}
        >
          Ingest Document
        </button>
        <button
          type="button"
          className={screen === "query" ? "active" : ""}
          onClick={() => setScreen("query")}
        >
          Query
        </button>
      </div>

      {screen === "query" ? <QueryScreen /> : <IngestScreen />}
    </div>
  );
}

export default App;
