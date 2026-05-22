import { useEffect, useRef, useState } from "react";
import { fetchConfig, streamChat, type WidgetConfig } from "./api";

interface Message {
  role: "user" | "assistant";
  content: string;
}

// widget_id + api_base come from the embed page's mount-div data attributes (set by the API embed
// route), falling back to the iframe URL query string for standalone use.
function readParams(): { widgetId: string | null; apiBase: string } {
  const root =
    document.getElementById("maintainers-copilot-root") ?? document.getElementById("root");
  const params = new URLSearchParams(window.location.search);
  return {
    widgetId: root?.getAttribute("data-widget-id") ?? params.get("widget_id"),
    apiBase:
      root?.getAttribute("data-api-base") ?? params.get("api_base") ?? window.location.origin,
  };
}

// Tell the host page our content height so it can resize the iframe (validated by origin there).
function useResizeBroadcast(ref: React.RefObject<HTMLDivElement>): void {
  useEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver(() => {
      const height = ref.current?.scrollHeight ?? 0;
      window.parent.postMessage({ type: "maintainers-copilot:resize", height }, "*");
    });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [ref]);
}

export function Widget(): JSX.Element {
  const { widgetId, apiBase } = readParams();
  const [config, setConfig] = useState<WidgetConfig | null>(null);
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionId = useRef<string | undefined>(undefined);
  const rootRef = useRef<HTMLDivElement>(null);
  useResizeBroadcast(rootRef);

  useEffect(() => {
    if (!widgetId) {
      setError("Missing widget_id.");
      return;
    }
    fetchConfig(apiBase, widgetId)
      .then(setConfig)
      .catch(() => setError("Could not load widget configuration."));
  }, [widgetId, apiBase]);

  async function send(): Promise<void> {
    const text = input.trim();
    if (!text || busy || !widgetId) return;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setBusy(true);
    setError(null);
    let answer = "";
    setMessages((prev) => [...prev, { role: "assistant", content: "…" }]);
    try {
      for await (const event of streamChat(apiBase, widgetId, {
        message: text,
        session_id: sessionId.current,
      })) {
        if (event.event === "session") {
          sessionId.current = String(event.data.session_id ?? "");
        } else if (event.event === "token" || event.event === "done") {
          answer = String(event.data.text ?? answer);
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = { role: "assistant", content: answer || "…" };
            return next;
          });
        } else if (event.event === "error") {
          setError(String(event.data.message ?? "Something went wrong."));
        }
      }
    } catch {
      setError("The assistant is unavailable right now.");
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setBusy(false);
    }
  }

  if (error && !config) {
    return <div ref={rootRef} className="mc-error">{error}</div>;
  }
  if (!config) {
    return <div ref={rootRef} className="mc-loading">Loading…</div>;
  }

  const accent = { ["--mc-accent" as string]: config.primary_color } as React.CSSProperties;

  return (
    <div ref={rootRef} className={`mc-widget mc-${config.theme}`} style={accent}>
      {!open && (
        <button className="mc-bubble" onClick={() => setOpen(true)} aria-label="Open chat">
          💬
        </button>
      )}
      {open && (
        <div className="mc-panel">
          <header className="mc-header">
            <span>Maintainer's Copilot</span>
            <button onClick={() => setOpen(false)} aria-label="Close chat">×</button>
          </header>
          <div className="mc-messages">
            {messages.length === 0 && <div className="mc-greeting">{config.greeting}</div>}
            {messages.map((message, index) => (
              <div key={index} className={`mc-msg mc-msg-${message.role}`}>
                {message.content}
              </div>
            ))}
            {error && <div className="mc-error">{error}</div>}
          </div>
          <div className="mc-input">
            <input
              value={input}
              placeholder="Ask about an issue…"
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && send()}
              disabled={busy}
            />
            <button onClick={send} disabled={busy}>Send</button>
          </div>
        </div>
      )}
    </div>
  );
}
