import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// JSON request helper for the evaluation endpoints: throws an Error with the backend's message.
async function apiJson(path, options) {
  const res = await fetch(`${API_BASE_URL}${path}`, options);
  const data = res.status === 204 ? null : await res.json().catch(() => null);
  if (!res.ok) throw new Error((data && data.detail) || `Request failed (HTTP ${res.status})`);
  return data;
}

// Wait for a background job ("generate" or "score") to finish, reporting its progress.
async function pollJob(kind, onProgress) {
  while (true) {
    const status = await apiJson(`/api/evaluation/${kind}/status`);
    onProgress(status);
    if (status.status === "done") return;
    await sleep(1000);
  }
}

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

function InspectionMetrics({ ttft, responseTime }) {
  return (
    <div className="metrics">
      <div className="metric-box">
        <label>TTFT (s)</label>
        <input type="text" readOnly value={ttft !== null ? ttft.toFixed(1) : ""} />
      </div>
      <div className="metric-box">
        <label>Total response time (s)</label>
        <input type="text" readOnly value={responseTime !== null ? responseTime.toFixed(1) : ""} />
      </div>
    </div>
  );
}

// Everything needed to turn a question, its expected answer and some chunks into a ground truth
// case. Used by the Query and Agent tabs.
function useGroundTruth() {
  const [mode, setMode] = useState(false); // whether the inspection panel is used to work on ground truth
  const [selectedIds, setSelectedIds] = useState(new Set());
  const [question, setQuestion] = useState(""); // the question of the case; editable, as a follow-up isn't a standalone question
  const [expectedAnswer, setExpectedAnswer] = useState("");
  const [replacesId, setReplacesId] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [savedId, setSavedId] = useState(null);
  const [activeCases, setActiveCases] = useState([]);

  const loadActiveCases = async () => {
    try {
      const data = await apiJson("/api/ground-truth");
      setActiveCases(data.cases.filter((c) => c.superseded_at === null));
    } catch (err) {
      setActiveCases([]);
    }
  };

  useEffect(() => {
    loadActiveCases();
  }, []);

  const reset = (nextQuestion = "", nextAnswer = "") => {
    setMode(false);
    setSelectedIds(new Set());
    setQuestion(nextQuestion);
    setExpectedAnswer(nextAnswer);
    setReplacesId("");
    setError(null);
    setSavedId(null);
  };

  const toggleChunk = (chunkId) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(chunkId)) next.delete(chunkId);
      else next.add(chunkId);
      return next;
    });
  };

  // chunks: every chunk on screen ({id, text, ...}); the selected ones become the expected chunks.
  const save = async (chunks) => {
    if (saving) return;

    setError(null);
    setSavedId(null);

    if (!question.trim()) {
      setError("Question must not be empty.");
      return;
    }
    if (!expectedAnswer.trim()) {
      setError("Expected answer must not be empty.");
      return;
    }
    if (selectedIds.size === 0) {
      setError("Select at least one chunk that supports the answer.");
      return;
    }

    setSaving(true);

    const expectedChunks = chunks
      .filter((c) => selectedIds.has(c.id))
      .map((c) => ({ chunk_id: c.id, chunk: c.text }));

    try {
      const data = await apiJson("/api/ground-truth", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          expected_answer: expectedAnswer,
          expected_chunks: expectedChunks,
          replaces_id: replacesId === "" ? null : Number(replacesId),
        }),
      });
      setSavedId(data.id);
      setReplacesId("");
      loadActiveCases();
    } catch (err) {
      setError(err.message || "Could not reach the backend. Is it running?");
    } finally {
      setSaving(false);
    }
  };

  return {
    mode, setMode, selectedIds, toggleChunk, question, setQuestion, expectedAnswer, setExpectedAnswer,
    replacesId, setReplacesId, saving, error, savedId, activeCases, reset, save,
  };
}

function GroundTruthToggle({ gt }) {
  return (
    <label className="gt-toggle">
      <input type="checkbox" checked={gt.mode} onChange={(e) => gt.setMode(e.target.checked)} />
      Work on ground truth
    </label>
  );
}

function GroundTruthForm({ gt, onSave }) {
  return (
    <div className="gt-form">
      <label htmlFor="gt-replaces">Save as</label>
      <select id="gt-replaces" value={gt.replacesId} onChange={(e) => gt.setReplacesId(e.target.value)}>
        <option value="">New case</option>
        {gt.activeCases.map((c) => (
          <option key={c.id} value={c.id}>
            Change #{c.id}: {c.question.length > 60 ? `${c.question.slice(0, 60)}…` : c.question}
          </option>
        ))}
      </select>

      <label htmlFor="gt-question">Question</label>
      <textarea
        id="gt-question"
        rows={2}
        value={gt.question}
        onChange={(e) => gt.setQuestion(e.target.value)}
      />

      <label htmlFor="gt-answer">Expected answer</label>
      <textarea
        id="gt-answer"
        rows={4}
        value={gt.expectedAnswer}
        onChange={(e) => gt.setExpectedAnswer(e.target.value)}
      />

      <p className="gt-hint">
        Select the retrieved chunks below that support the expected answer ({gt.selectedIds.size} selected).
      </p>

      <button type="button" onClick={onSave} disabled={gt.saving}>
        {gt.saving ? "Saving..." : "Save Ground Truth Case"}
      </button>
      {gt.error && <div className="error-box">{gt.error}</div>}
      {gt.savedId !== null && <div className="success-box">Saved as ground truth case #{gt.savedId}.</div>}
    </div>
  );
}

function ChunkList({ chunks, gt }) {
  return (
    <ul className="chunk-list">
      {chunks.map((chunk) => (
        <li key={chunk.id}>
          <label className={`chunk-item ${gt.mode ? "" : "chunk-item-plain"}`}>
            {gt.mode && (
              <input
                type="checkbox"
                checked={gt.selectedIds.has(chunk.id)}
                onChange={() => gt.toggleChunk(chunk.id)}
              />
            )}
            <span className="chunk-meta">
              {chunk.id} (score {chunk.score.toFixed(2)}
              {chunk.topic ? `, ${chunk.topic}` : ""})
            </span>
            <span className="chunk-text">{chunk.text}</span>
          </label>
        </li>
      ))}
    </ul>
  );
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
  const gt = useGroundTruth();

  const resetChunkBrowser = () => {
    setChunks([]);
    gt.reset();
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
      if (mode === "ask") gt.reset(inputText, data.answer.text);

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
      </div>

      <details className="inspection-panel">
        <summary>Inspection</summary>
        <InspectionMetrics ttft={ttft} responseTime={responseTime} />

        {mode === "ask" && chunks.length > 0 && <GroundTruthToggle gt={gt} />}

        {mode === "ask" && gt.mode && <GroundTruthForm gt={gt} onSave={() => gt.save(chunks)} />}

        {mode === "ask" && chunks.length > 0 && (
          <div className="chunk-browser">
            <label>Retrieved chunks ({chunks.length})</label>
            <ChunkList chunks={chunks} gt={gt} />
          </div>
        )}
      </details>
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
    if (loading || extracting) return;

    if (!documentId.trim()) {
      setError("Document ID must not be empty.");
      return;
    }
    if (!topic.trim()) {
      setError("Topic must not be empty.");
      return;
    }
    if (!text.trim()) {
      setError("Text must not be empty.");
      return;
    }

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

function AgentScreen() {
  const [question, setQuestion] = useState("");
  const [sessionId, setSessionId] = useState(null); // the chat being continued; null before the first question
  const [turns, setTurns] = useState([]); // [{question, answer}]
  const [loading, setLoading] = useState(false);
  const [sources, setSources] = useState([]); // cumulative over the chat; [n] in an answer is sources[n-1]
  const [steps, setSteps] = useState([]); // the latest turn's tool calls
  const [ttft, setTtft] = useState(null);
  const [responseTime, setResponseTime] = useState(null);
  const [error, setError] = useState(null);
  const gt = useGroundTruth();

  // Every distinct chunk the latest turn's searches returned, for the ground truth case.
  const allChunks = [...new Map(steps.flatMap((s) => s.chunks).map((c) => [c.id, c])).values()];

  const handleNewChat = () => {
    if (loading) return;
    setSessionId(null);
    setTurns([]);
    setSources([]);
    setSteps([]);
    setTtft(null);
    setResponseTime(null);
    setError(null);
    gt.reset();
  };

  const handleSubmit = async () => {
    const asked = question.trim();
    if (!asked || loading) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/agent/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: asked, session_id: sessionId }),
      });
      const data = await res.json().catch(() => null);

      if (!res.ok) {
        setError((data && data.detail) || `Request failed (HTTP ${res.status})`);
        return;
      }

      setSessionId(data.session_id);
      setTurns((prev) => [...prev, { question: asked, answer: data.answer.text }]);
      setSources(data.answer.sources);
      setSteps(data.steps);
      setTtft(data.ttft_seconds);
      setResponseTime(data.response_time_seconds);
      setQuestion("");
      gt.reset(asked, data.answer.text);
    } catch (err) {
      setError("Could not reach the backend. Is it running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {turns.length > 0 && (
        <div className="chat-transcript">
          {turns.map((turn, i) => (
            <div key={i} className="chat-turn">
              <div className="chat-question">{turn.question}</div>
              <div className="chat-answer">
                <ReactMarkdown>{turn.answer}</ReactMarkdown>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="ask-bar">
        <input
          type="text"
          placeholder={turns.length === 0 ? "Ask the agent a question..." : "Ask a follow-up question..."}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
        />
        <button type="button" onClick={handleSubmit} disabled={loading}>
          {loading ? "Waiting for the agent..." : "Submit"}
        </button>
        {turns.length > 0 && (
          <button type="button" className="secondary" onClick={handleNewChat} disabled={loading}>
            New chat
          </button>
        )}
      </div>

      {error && <div className="error-box">{error}</div>}

      {sources.length > 0 && (
        <ol className="sources-list">
          {sources.map((source, i) => (
            <li key={source.id}>
              <span className="source-id">
                {source.document_id} (chunk {source.chunk_id})
              </span>
              <span className="source-snippet"> — {source.snippet}</span>
            </li>
          ))}
        </ol>
      )}

      <details className="inspection-panel">
        <summary>Inspection</summary>
        <InspectionMetrics ttft={ttft} responseTime={responseTime} />

        {steps.length > 0 && <GroundTruthToggle gt={gt} />}

        {gt.mode && <GroundTruthForm gt={gt} onSave={() => gt.save(allChunks)} />}

        {steps.length > 0 && (
          <div className="agent-steps">
            <label>Agent steps of the latest answer ({steps.length})</label>
            {steps.map((step, i) => (
              <details key={i} className="agent-step">
                <summary>
                  [{i + 1}] {step.tool} — {JSON.stringify(step.arguments)} ({step.chunks.length} chunks)
                </summary>
                <ChunkList chunks={step.chunks} gt={gt} />
              </details>
            ))}
          </div>
        )}
      </details>
    </>
  );
}

function formatDate(iso) {
  return iso ? new Date(iso).toLocaleString() : "never";
}

function CorpusScreen() {
  const [documents, setDocuments] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/corpus`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to load corpus.");
        setDocuments(data.documents);
      } catch (err) {
        setError(err.message);
      }
    };
    load();
  }, []);

  if (error) return <div className="error-box">{error}</div>;
  if (!documents) return <p>Loading…</p>;
  if (documents.length === 0) return <p>No documents ingested yet.</p>;

  const byTopic = {};
  for (const doc of documents) (byTopic[doc.topic] ??= []).push(doc);

  return (
    <div className="corpus-tree">
      {Object.entries(byTopic).map(([topic, docs]) => (
        <details key={topic} open>
          <summary>
            {topic} <span className="corpus-count">({docs.length})</span>
          </summary>
          <ul>
            {docs.map((doc) => (
              <li key={doc.document_id} className={doc.sync_needed ? "sync-needed" : ""}>
                <span className="corpus-doc-id">{doc.document_id}</span>
                <span className="corpus-dates">
                  created {formatDate(doc.created_at)} · last synced {formatDate(doc.synced_at)}
                  {doc.sync_needed && " · sync needed"}
                </span>
              </li>
            ))}
          </ul>
        </details>
      ))}
    </div>
  );
}

const EVAL_METRIC_ROWS = [
  { label: "Cases", key: "case_count", format: (v) => v },
  { label: "Missing cases", key: "missing_case_count", format: (v) => v },
  { label: "Failed", key: "failed_count", format: (v) => v },
  { label: "Faithfulness", key: "avg_faithfulness", format: (v) => v.toFixed(2) },
  { label: "Correctness", key: "avg_correctness", format: (v) => v.toFixed(2) },
  { label: "Retrieval recall", key: "avg_retrieval_recall", format: (v) => v.toFixed(2) },
  { label: "Retrieval precision", key: "avg_retrieval_precision", format: (v) => v.toFixed(2) },
  { label: "Retrieval F1", key: "avg_retrieval_f1", format: (v) => v.toFixed(2) },
  { label: "Avg tokens", key: "avg_tokens_used", format: (v) => Math.round(v) },
  { label: "Avg response time (s)", key: "avg_response_time_seconds", format: (v) => v.toFixed(1) },
  { label: "Avg searches", key: "avg_searches", format: (v) => v.toFixed(1) },
];

function EvalsScreen() {
  const [runs, setRuns] = useState(null);
  const [gtCases, setGtCases] = useState([]);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(null); // null, or what is running: "Generating answers" | "Scoring"
  const [progress, setProgress] = useState({ total: 0, done: 0, failed: 0 });

  const [pipeline, setPipeline] = useState("rag");
  const [runLabel, setRunLabel] = useState("");
  const [repeats, setRepeats] = useState(1);
  const [topK, setTopK] = useState(5);
  const [maxLlmCalls, setMaxLlmCalls] = useState(10);
  const [searchTopK, setSearchTopK] = useState(5);

  const loadAll = async () => {
    try {
      const [runsData, gtData] = await Promise.all([
        apiJson("/api/evaluation/runs"),
        apiJson("/api/ground-truth"),
      ]);
      setRuns(runsData.runs);
      setGtCases(gtData.cases);
    } catch (err) {
      setError(err.message);
    }
  };

  useEffect(() => {
    loadAll();
  }, []);

  const runJobStep = async (phase, startPath, kind) => {
    setBusy(phase);
    setProgress({ total: 0, done: 0, failed: 0 });
    const started = await apiJson(startPath, { method: "POST" });
    if (started && started.started === false) {
      throw new Error("Another evaluation job is already running. Try again shortly.");
    }
    await pollJob(kind, setProgress);
  };

  const execute = async (steps) => {
    if (busy) return;
    setError(null);
    try {
      await steps();
    } catch (err) {
      setError(err.message || "Could not reach the backend. Is it running?");
    } finally {
      setBusy(null);
      loadAll();
    }
  };

  const generateAndScore = async (evalRunId) => {
    await runJobStep("Generating answers", `/api/evaluation/runs/${evalRunId}/generate`, "generate");
    await runJobStep("Scoring", `/api/evaluation/runs/${evalRunId}/score?rescore_all=false`, "score");
  };

  const handleStartRun = () =>
    execute(async () => {
      const isPositiveInt = (v) => Number.isInteger(v) && v >= 1;
      if (!isPositiveInt(repeats)) throw new Error("Repeats must be a whole number of at least 1.");
      let config;
      if (pipeline === "rag") {
        if (!isPositiveInt(topK)) throw new Error("Top K must be a whole number of at least 1.");
        config = { pipeline, top_k: topK };
      } else {
        if (!Number.isInteger(maxLlmCalls) || maxLlmCalls < 2) {
          throw new Error("Max model calls must be a whole number of at least 2.");
        }
        if (!isPositiveInt(searchTopK)) throw new Error("Chunks per search must be a whole number of at least 1.");
        config = { pipeline, max_llm_calls: maxLlmCalls, search_top_k: searchTopK };
      }

      setBusy("Generating answers");
      setProgress({ total: 0, done: 0, failed: 0 });
      const created = await apiJson("/api/evaluation/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: runLabel.trim() || null, repeats, config }),
      });
      await pollJob("generate", setProgress);
      await runJobStep("Scoring", `/api/evaluation/runs/${created.eval_run_id}/score?rescore_all=false`, "score");
    });

  const handleUpdateRun = (evalRunId) => execute(() => generateAndScore(evalRunId));

  const handleRescore = (evalRunId) =>
    execute(() => runJobStep("Scoring", `/api/evaluation/runs/${evalRunId}/score?rescore_all=true`, "score"));

  const handleSupersede = (gtCase) => {
    if (!window.confirm(`Mark case #${gtCase.id} as outdated? Evaluation runs will stop counting it.`)) return;
    execute(() => apiJson(`/api/ground-truth/${gtCase.id}/supersede`, { method: "POST" }));
  };

  if (!runs) return error ? <div className="error-box">{error}</div> : <p>Loading…</p>;

  const activeCases = gtCases.filter((c) => c.superseded_at === null);

  return (
    <>
      <details className="eval-form">
        <summary>New evaluation run</summary>
        <div className="eval-form-fields">
          <label>
            Pipeline
            <select value={pipeline} onChange={(e) => setPipeline(e.target.value)}>
              <option value="rag">Plain RAG</option>
              <option value="agent">Agent</option>
            </select>
          </label>
          {pipeline === "rag" ? (
            <label>
              Top K (chunks retrieved)
              <input type="number" min="1" value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
            </label>
          ) : (
            <>
              <label>
                Max model calls
                <input
                  type="number"
                  min="2"
                  value={maxLlmCalls}
                  onChange={(e) => setMaxLlmCalls(Number(e.target.value))}
                />
              </label>
              <label>
                Chunks per search
                <input
                  type="number"
                  min="1"
                  value={searchTopK}
                  onChange={(e) => setSearchTopK(Number(e.target.value))}
                />
              </label>
            </>
          )}
          <label>
            Repeats per case
            <input type="number" min="1" value={repeats} onChange={(e) => setRepeats(Number(e.target.value))} />
          </label>
          <label>
            Label (optional)
            <input type="text" value={runLabel} onChange={(e) => setRunLabel(e.target.value)} />
          </label>
          <button type="button" onClick={handleStartRun} disabled={busy !== null}>
            Start run
          </button>
        </div>
      </details>

      {busy && (
        <p className="eval-progress">
          {busy}: {progress.done}/{progress.total}
          {progress.failed > 0 && ` (${progress.failed} failed)`}
        </p>
      )}

      {error && <div className="error-box">{error}</div>}

      {runs.length === 0 ? (
        <p>No evaluation runs yet.</p>
      ) : (
        <div className="evals-table-wrapper">
          <table className="evals-table">
            <thead>
              <tr>
                <th />
                {runs.map((run) => (
                  <th key={run.id}>
                    <div>
                      #{run.id} · {run.pipeline}
                    </div>
                    <div className="evals-run-meta">{new Date(run.created_at).toLocaleString()}</div>
                    <div className="evals-run-meta">{run.status}</div>
                    <div className="evals-run-meta">
                      {Object.entries(run.config).map(([k, v]) => `${k}=${v}`).join(", ")}
                    </div>
                    {run.label && <div className="evals-run-meta">{run.label}</div>}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {EVAL_METRIC_ROWS.map(({ label, key, format }) => (
                <tr key={key}>
                  <th scope="row">{label}</th>
                  {runs.map((run) => (
                    <td key={run.id}>{run[key] === null ? "–" : format(run[key])}</td>
                  ))}
                </tr>
              ))}
              <tr>
                <th scope="row">Actions</th>
                {runs.map((run) => (
                  <td key={run.id} className="evals-actions">
                    {(run.missing_case_count > 0 || run.failed_count > 0) && (
                      <button type="button" onClick={() => handleUpdateRun(run.id)} disabled={busy !== null}>
                        Update run
                      </button>
                    )}
                    <button type="button" onClick={() => handleRescore(run.id)} disabled={busy !== null}>
                      Re-score
                    </button>
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      )}

      <details className="gt-list">
        <summary>
          Ground truth cases ({activeCases.length} active
          {gtCases.length > activeCases.length && `, ${gtCases.length - activeCases.length} outdated`})
        </summary>
        <ul>
          {gtCases.map((c) => (
            <li key={c.id} className={c.superseded_at ? "gt-outdated" : ""}>
              <span className="gt-question">
                #{c.id} {c.question}
              </span>
              <span className="gt-meta">
                {c.superseded_at
                  ? `outdated${c.superseded_by ? `, replaced by #${c.superseded_by}` : ""}`
                  : `${c.expected_chunk_count} expected chunk${c.expected_chunk_count === 1 ? "" : "s"}`}
              </span>
              {!c.superseded_at && (
                <button type="button" onClick={() => handleSupersede(c)} disabled={busy !== null}>
                  Mark outdated
                </button>
              )}
            </li>
          ))}
        </ul>
      </details>
    </>
  );
}

function AdminScreen() {
  const [syncLoading, setSyncLoading] = useState(false);
  const [syncError, setSyncError] = useState(null);
  const [syncResult, setSyncResult] = useState(null);

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
          className={screen === "corpus" ? "active" : ""}
          onClick={() => setScreen("corpus")}
        >
          Corpus
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
          className={screen === "agent" ? "active" : ""}
          onClick={() => setScreen("agent")}
        >
          Agent
        </button>
        <button
          type="button"
          className={screen === "evals" ? "active" : ""}
          onClick={() => setScreen("evals")}
        >
          Evals
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
      {screen === "agent" && <AgentScreen />}
      {screen === "corpus" && <CorpusScreen />}
      {screen === "evals" && <EvalsScreen />}
      {screen === "admin" && <AdminScreen />}
    </div>
  );
}

export default App;
