// Backend client for the embedded widget: public config + SSE chat stream (over POST).

export interface WidgetConfig {
  widget_id: string;
  theme: "light" | "dark";
  primary_color: string;
  position: string;
  greeting: string;
  enabled_tools: string[];
  is_active: boolean;
}

export interface SSEMessage {
  event: string;
  data: Record<string, unknown>;
}

export async function fetchConfig(apiBase: string, widgetId: string): Promise<WidgetConfig> {
  const response = await fetch(`${apiBase}/widgets/${encodeURIComponent(widgetId)}/config`);
  if (!response.ok) {
    throw new Error(`config request failed: ${response.status}`);
  }
  return (await response.json()) as WidgetConfig;
}

// Consume the backend SSE stream. /chat/stream is a POST endpoint, so EventSource (GET-only) can't be
// used; we read the response body and parse the `event:`/`data:` frames ourselves.
export async function* streamChat(
  apiBase: string,
  body: { message: string; conversation_id?: string },
  token?: string,
): AsyncGenerator<SSEMessage> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const response = await fetch(`${apiBase}/chat/stream`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  if (!response.ok || !response.body) {
    throw new Error(`chat request failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) {
        yield { event, data: JSON.parse(data) as Record<string, unknown> };
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}
