import { useState } from "react";

function App() {
  const [question, setQuestion] = useState("");

  return (
    <div className="app">
      <h1>reportagent</h1>
      <div className="ask-bar">
        <input
          type="text"
          placeholder="Ask a question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button type="button">Ask</button>
      </div>
    </div>
  );
}

export default App;
