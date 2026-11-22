#!/usr/bin/env python3

"""
bookwork

run: MODEL="available model" python3 server.py
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
import re

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("MODEL", "")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
INDEX_HTML = Path(__file__).with_name("index.html")
MAX_CACHE = 2000

SYSTEM_PROMPT = """You are a German grammar and vocabulary tutor. A reader clicked a word or phrase in a German text.
Answer with a JSON object with these fields:
- part_of_speech: noun, verb, adjective, adverb, pronoun, preposition, conjunction, article or other.
- basic_form: nouns: article + nominative singular ("der Himmel"). Verbs: infinitive, including a separable prefix found elsewhere in the sentence ("anrufen"). Other words: dictionary form.
- plural: nouns only: "die" + nominative plural ("die Himmel", "die Flugzeuge"). Many nouns change in the plural (-e, -en, -er, -s, umlaut), some stay the same. Empty string for non-nouns.
- grammar: the form as used in this sentence, e.g. "accusative singular, direct object of 'sahen'" or "simple past, 3rd person plural".
- translation: the best English translation in this context.
- explanation: exactly one short English sentence explaining the meaning in this context; mention idioms if applicable.
Follow German spelling and capitalization rules."""

FEW_SHOT = [
    ("Die Flugzeuge landeten pünktlich in Zürich.", "Flugzeuge", {
        "part_of_speech": "noun", "basic_form": "das Flugzeug", "plural": "die Flugzeuge",
        "grammar": "nominative plural, subject of 'landeten'",
        "translation": "airplanes",
        "explanation": "It refers to the airplanes that landed on time in Zurich."}),
    ("Sie sahen den Hund im Garten.", "den Hund", {
        "part_of_speech": "noun", "basic_form": "der Hund", "plural": "die Hunde",
        "grammar": "accusative singular, direct object of 'sahen'",
        "translation": "the dog",
        "explanation": "The dog is what they saw in the garden."}),
    ("Er gab dem Mädchen ein Buch.", "dem Mädchen", {
        "part_of_speech": "noun", "basic_form": "das Mädchen", "plural": "die Mädchen",
        "grammar": "dative singular, indirect object of 'gab'",
        "translation": "the girl",
        "explanation": "The girl is the one receiving the book."}),
    ("Er rief seine Mutter gestern Abend an.", "rief", {
        "part_of_speech": "verb", "basic_form": "anrufen", "plural": "",
        "grammar": "simple past, 3rd person singular; separable verb, prefix 'an' at the end",
        "translation": "called (on the phone)",
        "explanation": "He phoned his mother yesterday evening."}),
    ("Die Kinder waren über den lauten Knall erschrocken.", "erschrocken", {
        "part_of_speech": "verb", "basic_form": "erschrecken", "plural": "",
        "grammar": "past participle, used with 'waren' to describe a state",
        "translation": "frightened",
        "explanation": "It describes how the loud bang startled the children."}),
]

SCHEMA = {
    "type": "object",
    "properties": {
        "part_of_speech": {"type": "string", "enum": [
            "noun", "verb", "adjective", "adverb", "pronoun",
            "preposition", "conjunction", "article", "other"]},
        "basic_form": {"type": "string"},
        "plural": {"type": "string"},
        "grammar": {"type": "string"},
        "translation": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["part_of_speech", "basic_form", "plural", "grammar", "translation", "explanation"],
}

def build_user_message(word, sentence, context):
    msg = f"Sentence: {sentence}\nWord or phrase to explain: {word}"
    if context and context != sentence:
        msg += f"\n\nSurrounding text (extra context only):\n{context}"
    return msg


def build_messages(word, sentence, context):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for s, w, answer in FEW_SHOT:
        msgs.append({"role": "user", "content": build_user_message(w, s, "")})
        msgs.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
    msgs.append({"role": "user", "content": build_user_message(word, sentence, context)})
    return msgs


def ask_ollama(model, messages):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "format": SCHEMA,
        "keep_alive": "30m",
        "options": {"temperature": 0},
    }
    for attempt in (1, 2):
        req = urllib.request.Request(
            OLLAMA_URL + "/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                content = json.load(r)["message"]["content"]
            return json.loads(content)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if attempt == 1 and "think" in body.lower():
                payload.pop("think", None)
                continue
            raise RuntimeError(f"Ollama returned {e.code}: {body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_URL}: {e.reason}")
        except (ValueError, KeyError) as e:
            raise RuntimeError(f"Model returned invalid JSON: {e}")


def format_answer(a):
    basic = a.get("basic_form", "").strip()
    line = f"Basic form: {basic}"
    if a.get("part_of_speech") == "noun":
        plural = a.get("plural", "").strip()
        if plural.lower().startswith(("der ", "die ", "das ")):
            plural = plural[4:]
        line += f" - die {plural}" if plural else " - (no plural)"
    explanation = a.get("explanation", "").strip()
    first = re.match(r".*?[.!?](?=\s|$)", explanation, re.S)
    if first:
        explanation = first.group(0)
    return "\n".join([
        line,
        f"Grammar: {a.get('grammar', '').strip()}",
        f"Translation: {a.get('translation', '').strip()}",
        f"In this sentence: {explanation}",
    ])

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
        "options": {"temperature": 0},
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

    def send(self, status, content_type, data):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status, obj):
        self.send(status, "application/json; charset=utf-8", json.dumps(obj).encode())

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            try:
                self.send(200, "text/html; charset=utf-8", INDEX_HTML.read_bytes())
            except OSError as e:
                self.send_json(500, {"error": f"Cannot read {INDEX_HTML}: {e}"})
        elif path == "/api/models":
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
            text = _cache.get(key)
        if text is None:
            try:
                text = format_answer(ask_ollama(model, build_messages(word, sentence, context)))
            except Exception as e:
                return self.send_json(502, {"error": str(e)})
            with _cache_lock:
                if len(_cache) >= MAX_CACHE:
                    _cache.clear()
                _cache[key] = text
        self.send(200, "text/plain; charset=utf-8", text.encode())

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