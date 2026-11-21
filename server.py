#!/usr/bin/env python3

"""
bookwork

run: python3 server.py
open: http://localhost:8000
"""

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("MODEL", "")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
TARGET_LANG = os.environ.get("TARGET_LANG", "English")
INDEX_HTML = Path(__file__).with_name("index.html")
MAX_CACHE = 2000

SYSTEM_PROMPT = f"""You are a language tutor helping a reader to understand a text
in a foreign language. The learner clicked a word they do not understand. Explain it 
in English. Be concise and accurate. Answer in plain text, no markdown, using exactly these lines:
Base form: <dictionary word form , only if a nouns include article/gender. make sure to catch suffix and prefixes which in German sometimes appear elsewhere in the sentence. >
Translation: <best English translation in the context where the word appears in the text>
In this sentence: <In 1 sentence: Explain the word in its context; mention idioms if applicable. Keep it short and precise, 1 sentence is preferable to 2.>
"""

_cache = {}
_cache_lock = threading.Lock()

def list_models():
    with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=10) as r:
        return [m["name"] for m in json.load(r).get("models", [])]
    
def pick_default(models):
    if DEFAULT_MODEL:
        return DEFAULT_MODEL
    for m in models:
        if "qwen" in m.lower():
            return m
    return models[0] if models else ""

def open_chat_stream(model, messages):
    payload = {
        "model": model, 
        "messages": messages,
        "stream": True,
        "think": False, 
        "keep_alive": "30m",
        "options": {"temperature": 0.2},
    }
    for attempt in (1,2):
        req = urllib.request.Request(
            OLLAMA_URL + "/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            return urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if attempt == 1 and "think" in body.lower():
                payload.pop("think", None)
                continue
            raise RuntimeError(f"Ollama returned {e.code}: {body}")
        except urllib.error.URLError as r:
            raise RuntimeError(f"Cannot reach Ollama ar {OLLAMA_URL}: {e.reason} ")
        
def build_user_message(word, sentence, context):
    msg = f"Sentence: {sentence}\nWord or phrase to explain: {word}"
    if context and context != sentence:
        msg += f"\n\nSurrounding text (extra context only: \n{context})"
    return msg

class Handle(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def send_json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def start_text_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")  # in case a proxy sits in front later
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            try:
                data = INDEX_HTML.read_bytes()
            except OSError as e:
                return self.send_json(500, {"error": f"Cannot read {INDEX_HTML}: {e}"})
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/api/models":
            try:
                models = list_models()
                self.send_json(200, {"models": models, "default": pick_default(models)})
            except Exception as e:
                self.send_json(502, {"error": f"Cannot reach Ollama at {OLLAMA_URL}: {e}"})
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/explain":
            return self.send_json(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            word = str(body.get("word", "")).strip()[:200]
            sentence = str(body.get("sentence", "")).strip()[:1500]
            context = str(body.get("context", "")).strip()[:3000]
            model = str(body.get("model") or DEFAULT_MODEL).strip()
        except (ValueError, TypeError):
            return self.send_json(400, {"error": "Invalid JSON"})
        if not word:
            return self.send_json(400, {"error": "No word given"})
        if not model:
            try:
                model = pick_default(list_models())
            except Exception as e:
                return self.send_json(502, {"error": str(e)})

        key = (model, word, sentence)
        with _cache_lock:
            cached = _cache.get(key)
        if cached is not None:
            self.start_text_stream()
            self.wfile.write(cached.encode())
            return

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(word, sentence, context)},
        ]
        try:
            upstream = open_chat_stream(model, messages)
        except Exception as e:
            return self.send_json(502, {"error": str(e)})

        self.start_text_stream()
        parts = []
        try:
            with upstream:
                for line in upstream:
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    if msg.get("error"):
                        self.wfile.write(f"\n[Ollama error: {msg['error']}]".encode())
                        return
                    chunk = msg.get("message", {}).get("content", "")
                    if chunk:
                        parts.append(chunk)
                        self.wfile.write(chunk.encode())
                        self.wfile.flush()
                    if msg.get("done"):
                        with _cache_lock:
                            if len(_cache) >= MAX_CACHE:
                                _cache.clear()
                            _cache[key] = "".join(parts)
                        return
        except (BrokenPipeError, ConnectionResetError):
            # The user clicked another word and the browser aborted this request.
            # Leaving the `with` block closes the Ollama connection, which stops generation.
            pass

def main():
    try:
        models = list_models()
        print(f"Ollama OK at {OLLAMA_URL}. Models: {', '.join(models) or '(none)'}")
        print(f"Default model: {pick_default(models) or '(none, run: ollama pull qwen2.5:7b)'}")
    except Exception as e:
        print(f"WARNING: cannot reach Ollama at {OLLAMA_URL} ({e}). Is `ollama serve` running?")
    server = ThreadingHTTPServer((HOST, PORT), Handle)
    print(f"Word Explainer running on http://localhost:{PORT} (listening on {HOST})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    sys.exit(main())