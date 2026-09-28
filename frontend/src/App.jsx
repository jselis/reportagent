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

  const [topN, setTopN] = useState(20);
  const [chunks, setChunks] = useState([]);
  const [selectedChunkIds, setSelectedChunkIds] = useState(new Set());
  const [gtAnswer, setGtAnswer] = useState("");
  const [gtSaving, setGtSaving] = useState(false);
  const [gtError, setGtError] = useState(null);
  const [gtSavedId, setGtSavedId] = useState(null);

  const resetChunkBrowser = () => {
    setChunks([]);
    setSelectedChunkIds(new Set());
    setGtAnswer("");
    setGtError(null);
    setGtSavedId(null);
  };

  const handleModeChange = (newMode) => {
    setMode(newMode);
    setResult("");
    setSources([]);
    setTtft(null);
    setResponseTime(null);
    setError(null);
    resetChunkBrowser();
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
    resetChunkBrowser();

    try {
      const askPromise = fetch(`${API_BASE_URL}${url}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [field]: inputText }),
      });

      const chunksPromise =
        mode === "ask"
          ? fetch(`${API_BASE_URL}/api/retrieve?${new URLSearchParams({ q: inputText, top_n: topN })}`)
          : null;

      const res = await askPromise;
      const data = await res.json();

      if (!res.ok) {
        setError(data.detail || `Request failed (HTTP ${res.status})`);
        return;
      }

      setResult(formatResult(mode, data));
      setSources(mode === "ask" ? data.answer.sources : []);
      setTtft(data.ttft_seconds);
      setResponseTime(data.response_time_seconds);
      if (mode === "ask") setGtAnswer(data.answer.text);

      if (chunksPromise) {
        const chunksRes = await chunksPromise;
        const chunksData = await chunksRes.json();
        if (chunksRes.ok) setChunks(chunksData);
      }
    } catch (err) {
      setError("Could not reach the backend. Is it running?");
    } finally {
      setLoading(false);
    }
  };

  const toggleChunkSelected = (chunkId) => {
    setSelectedChunkIds((prev) => {
      const next = new Set(prev);
      if (next.has(chunkId)) next.delete(chunkId);
      else next.add(chunkId);
      return next;
    });
  };

  const handleSaveGroundTruth = async () => {
    if (selectedChunkIds.size === 0 || !gtAnswer.trim() || gtSaving) return;

    setGtSaving(true);
    setGtError(null);
    setGtSavedId(null);

    const expectedChunks = chunks
      .filter((c) => selectedChunkIds.has(c.id))
      .map((c) => ({ chunk_id: c.id, chunk: c.text }));

    try {
      const res = await fetch(`${API_BASE_URL}/api/ground-truth`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: inputText,
          expected_answer: gtAnswer,
          expected_chunks: expectedChunks,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        setGtError(data.detail || `Request failed (HTTP ${res.status})`);
        return;
      }

      setGtSavedId(data.id);
    } catch (err) {
      setGtError("Could not reach the backend. Is it running?");
    } finally {
      setGtSaving(false);
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

      {mode === "ask" && (
        <div className="top-n-selector">
          <label htmlFor="top-n-input">Show top</label>
          <input
            id="top-n-input"
            type="number"
            min="1"
            max="100"
            value={topN}
            onChange={(e) => setTopN(Number(e.target.value))}
          />
          <label htmlFor="top-n-input">chunks (answer still uses top 5)</label>
        </div>
      )}

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

      {mode === "ask" && chunks.length > 0 && (
        <div className="chunk-browser">
          <label>Retrieved chunks ({chunks.length}) — select the ones that support the answer</label>
          <ul className="chunk-list">
            {chunks.map((chunk) => (
              <li key={chunk.id}>
                <label className="chunk-item">
                  <input
                    type="checkbox"
                    checked={selectedChunkIds.has(chunk.id)}
                    onChange={() => toggleChunkSelected(chunk.id)}
                  />
                  <span className="chunk-meta">
                    {chunk.id} (score {chunk.score.toFixed(2)}
                    {chunk.topic ? `, ${chunk.topic}` : ""})
                  </span>
                  <span className="chunk-text">{chunk.text}</span>
                </label>
              </li>
            ))}
          </ul>

          {selectedChunkIds.size > 0 && (
            <div className="gt-form">
              <label htmlFor="gt-answer">Expected answer for this ground truth case</label>
              <textarea
                id="gt-answer"
                rows={4}
                value={gtAnswer}
                onChange={(e) => setGtAnswer(e.target.value)}
              />
              <button type="button" onClick={handleSaveGroundTruth} disabled={gtSaving}>
                {gtSaving ? "Saving..." : "Save Ground Truth Case"}
              </button>
              {gtError && <div className="error-box">{gtError}</div>}
              {gtSavedId !== null && (
                <div className="success-box">Saved as ground truth case #{gtSavedId}.</div>
              )}
            </div>
          )}
        </div>
      )}
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
  const [batchResults, setBatchResults] = useState([]);
  const [batchRunning, setBatchRunning] = useState(false);

  const extractFileText = async (file) => {
    const isPdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
    if (!isPdf) return file.text();

    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API_BASE_URL}/api/extract-text`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Could not extract text from "${file.name}"`);
    return data.text;
  };

  const handleFolderChange = async (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = ""; // allow re-selecting the same folder later
    if (files.length === 0) return;

    if (!topic.trim()) {
      setError("Set a Topic before selecting a folder to batch-ingest.");
      return;
    }

    const qualifying = files.filter((f) => /\.(txt|pdf)$/i.test(f.name));
    if (qualifying.length === 0) {
      setError("No .txt or .pdf files found in that folder.");
      return;
    }

    setError(null);
    setSuccessMessage(null);
    setBatchRunning(true);
    setBatchResults(
      qualifying.map((f) => ({
        name: f.webkitRelativePath || f.name,
        status: "pending",
      }))
    );

    for (let i = 0; i < qualifying.length; i++) {
      const file = qualifying[i];
      const relPath = file.webkitRelativePath || file.name;
      const documentId = relPath.replace(/\.(txt|pdf)$/i, "");

      setBatchResults((prev) =>
        prev.map((r, idx) => (idx === i ? { ...r, status: "processing" } : r))
      );

      try {
        const fileText = await extractFileText(file);

        const res = await fetch(`${API_BASE_URL}/api/ingest`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ document_id: documentId, topic, text: fileText }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Ingest failed");

        setBatchResults((prev) =>
          prev.map((r, idx) =>
            idx === i
              ? { ...r, status: "done", message: `${data.chunks_ingested} chunk${data.chunks_ingested === 1 ? "" : "s"}` }
              : r
          )
        );
      } catch (err) {
        setBatchResults((prev) =>
          prev.map((r, idx) => (idx === i ? { ...r, status: "error", message: err.message } : r))
        );
      }
    }

    setBatchRunning(false);
  };

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

      <div className="file-upload">
        <label htmlFor="ingest-folder">
          Or select a folder to batch-ingest every .txt/.pdf file in it (using the Topic above,
          document IDs taken from filenames):
        </label>
        <input
          id="ingest-folder"
          type="file"
          webkitdirectory=""
          directory=""
          multiple
          onChange={handleFolderChange}
          disabled={extracting || batchRunning}
        />
      </div>
      {fileError && <div className="error-box">{fileError}</div>}

      <button type="button" onClick={handleSubmit} disabled={loading || extracting || batchRunning}>
        {loading ? "Ingesting..." : "Ingest"}
      </button>

      {error && <div className="error-box">{error}</div>}
      {successMessage && <div className="success-box">{successMessage}</div>}

      {batchResults.length > 0 && (
        <div className="batch-results">
          <label>
            Batch ingest — {batchResults.filter((r) => r.status === "done").length}/
            {batchResults.length} done
          </label>
          <ul className="batch-list">
            {batchResults.map((r, i) => (
              <li key={i} className={`batch-item batch-${r.status}`}>
                <span className="batch-name">{r.name}</span>
                <span className="batch-status">
                  {r.status === "pending" && "Waiting..."}
                  {r.status === "processing" && "Processing..."}
                  {r.status === "done" && `✓ ${r.message}`}
                  {r.status === "error" && `✗ ${r.message}`}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function AdminScreen() {
  const [syncLoading, setSyncLoading] = useState(false);
  const [syncError, setSyncError] = useState(null);
  const [syncResult, setSyncResult] = useState(null);

  const [evalLoading, setEvalLoading] = useState(false);
  const [evalError, setEvalError] = useState(null);
  const [evalResult, setEvalResult] = useState(null);

  const handleSync = async () => {
    if (syncLoading) return;

    setSyncLoading(true);
    setSyncError(null);
    setSyncResult(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/sync-metadata`, { method: "POST" });
      const data = await res.json();

      if (!res.ok) {
        setSyncError(data.detail || `Request failed (HTTP ${res.status})`);
        return;
      }

      setSyncResult(data);
    } catch (err) {
      setSyncError("Could not reach the backend. Is it running?");
    } finally {
      setSyncLoading(false);
    }
  };

  const handleEvaluate = async () => {
    if (evalLoading) return;

    setEvalLoading(true);
    setEvalError(null);
    setEvalResult(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/evaluate`, { method: "POST" });
      const data = await res.json();

      if (!res.ok) {
        setEvalError(data.detail || `Request failed (HTTP ${res.status})`);
        return;
      }

      setEvalResult(data);
    } catch (err) {
      setEvalError("Could not reach the backend. Is it running?");
    } finally {
      setEvalLoading(false);
    }
  };

  return (
    <div className="admin-panel">
      <p className="admin-description">
        If you corrected a document's metadata directly in the database, use this to push
        those changes to every chunk of the affected documents in Pinecone.
      </p>

      <button type="button" onClick={handleSync} disabled={syncLoading}>
        {syncLoading ? "Syncing..." : "Sync Metadata"}
      </button>

      {syncError && <div className="error-box">{syncError}</div>}

      {syncResult && (
        <div className="sync-result">
          <p>
            <strong>Synced ({syncResult.synced.length}):</strong>{" "}
            {syncResult.synced.length > 0 ? syncResult.synced.join(", ") : "nothing pending"}
          </p>
          {syncResult.failed.length > 0 && (
            <p className="sync-failed">
              <strong>Failed ({syncResult.failed.length}):</strong> {syncResult.failed.join(", ")}
            </p>
          )}
        </div>
      )}

      <hr className="admin-divider" />

      <p className="admin-description">
        Re-run every ground truth case's question through the current retrieval + answer
        pipeline, storing the result for comparison against the curated expected answer/chunks.
      </p>

      <button type="button" onClick={handleEvaluate} disabled={evalLoading}>
        {evalLoading ? "Running evaluation..." : "Run Evaluation"}
      </button>

      {evalError && <div className="error-box">{evalError}</div>}

      {evalResult && (
        <div className="sync-result">
          <p>
            <strong>Evaluated ({evalResult.evaluated.length}):</strong>{" "}
            {evalResult.evaluated.length > 0 ? evalResult.evaluated.join(", ") : "no cases found"}
          </p>
          {evalResult.failed.length > 0 && (
            <p className="sync-failed">
              <strong>Failed ({evalResult.failed.length}):</strong> {evalResult.failed.join(", ")}
            </p>
          )}
        </div>
      )}
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
        <button
          type="button"
          className={screen === "admin" ? "active" : ""}
          onClick={() => setScreen("admin")}
        >
          Admin
        </button>
      </div>

      {screen === "query" && <QueryScreen />}
      {screen === "ingest" && <IngestScreen />}
      {screen === "admin" && <AdminScreen />}
    </div>
  );
}

export default App;
