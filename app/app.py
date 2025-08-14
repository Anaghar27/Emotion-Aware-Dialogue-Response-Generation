import json
from datetime import datetime
from flask import Flask, Response, request, render_template, jsonify
from werkzeug.serving import WSGIRequestHandler
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from response_runtime import ResponseRuntime

# Initialize Flask app and runtime engine
app = Flask(__name__, static_folder="static", template_folder="templates")
runtime = ResponseRuntime()

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

def sse_pack(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

@app.post("/api/chat")
def chat():
    payload = request.get_json(force=True) or {}
    user_msg = (payload.get("message") or "").strip()
    history = payload.get("history") or []
    temperature = float(payload.get("temperature", 0.7))
    top_p = float(payload.get("top_p", 0.9))
    max_new_tokens = int(payload.get("max_new_tokens", 256))

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    # Append the new user message to history
    history = list(history) + [{"role": "user", "content": user_msg}]
    turns = [(m["role"], m["content"]) for m in history]

    # Classify emotion and dialogue act for the latest message
    emo, dact = runtime.engine.classify_last(turns)

    def stream():
        # Send classification metadata
        yield sse_pack("meta", {"emotion": emo, "act": dact})

        # Stream generated text chunks
        for chunk in runtime.generate(
            history=history,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        ):
            yield sse_pack("delta", {"text": chunk})
        yield sse_pack("done", {"text": ""})

    # SSE response headers for real-time streaming
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return Response(stream(), headers=headers)

@app.post("/api/export")
def export_chat():
    payload = request.get_json(force=True) or {}
    history = payload.get("history") or []
    export = {"exported_at": datetime.utcnow().isoformat() + "Z", "messages": history}
    name = f"chat_export_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    # Save the chat export to a file
    with open(name, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    return jsonify({"filename": name})

if __name__ == "__main__":
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)
