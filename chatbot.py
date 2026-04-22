#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════╗
║        NEW GEN AI Chat Bot — Server              ║
║        Author  : Anubhav                         ║
║        Engine  : Google Gemini 1.5 Flash         ║
║        Storage : JSON Knowledge Base             ║
╚══════════════════════════════════════════════════╝

Features:
  • TF-IDF cosine-similarity search over knowledge base
  • Auto-update: background thread resolves pending Qs
  • Answer enhancement: short answers expanded by Gemini
  • Pending store: unanswered Qs saved and answered later
  • Zero external web-framework — pure http.server
"""

import json
import os
import time
import threading
import numpy as np
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ═══════════════════════════ CONFIG ═══════════════════════════════════

GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
KB_FILE          = "knowledge_base.json"
SIMILARITY_TH    = 0.40          # cosine similarity threshold (0–1)
SHORT_ANSWER_LEN = 180           # answers shorter than this get enhanced
UPDATE_INTERVAL  = 1800          # auto-update every 30 minutes (seconds)
MAX_PENDING_RESOLVE = 5          # max pending Qs resolved per cycle
PORT             = 8080

# ─── Gemini setup ────────────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ═══════════════════════════ KNOWLEDGE BASE I/O ═══════════════════════

def load_kb() -> dict:
    """Load knowledge base from JSON file."""
    with open(KB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_kb(kb: dict):
    """Persist knowledge base to JSON file with updated timestamp."""
    kb["metadata"]["last_updated"] = datetime.now().isoformat()
    kb["metadata"]["total_qa"]     = len(kb["qa_pairs"])
    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

# ═══════════════════════════ SIMILARITY SEARCH ════════════════════════

def find_best_match(query: str, kb: dict) -> tuple:
    """
    TF-IDF cosine similarity search.
    Returns (matched_pair_or_None, similarity_score).
    """
    qa_pairs = kb.get("qa_pairs", [])
    if not qa_pairs:
        return None, 0.0

    questions = [pair["question"] for pair in qa_pairs]
    corpus    = questions + [query]

    # Build TF-IDF matrix and compute cosine similarity
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf      = vectorizer.fit_transform(corpus)
    scores     = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()

    best_idx   = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score >= SIMILARITY_TH:
        return qa_pairs[best_idx], best_score
    return None, best_score

# ═══════════════════════════ GEMINI HELPERS ═══════════════════════════

def call_gemini(prompt: str) -> str:
    """Send prompt to Gemini and return text response."""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        print(f"[Gemini error] {exc}")
        return ""


def generate_answer(question: str) -> str:
    """Ask Gemini for a fresh, comprehensive answer."""
    prompt = (
        "You are NEW GEN AI Chat Bot, created by Anubhav. "
        "Provide a clear, structured, and comprehensive answer with bullet points where helpful. "
        "Keep the tone friendly and informative.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return call_gemini(prompt)


def enhance_answer(question: str, current_answer: str) -> str:
    """Expand and improve an existing short or incomplete answer."""
    prompt = (
        "You are NEW GEN AI Chat Bot, created by Anubhav. "
        "The following answer is incomplete or too brief. "
        "Expand it with more detail, add bullet points, examples, and make it fully comprehensive. "
        "Return only the enhanced answer text.\n\n"
        f"Question: {question}\n"
        f"Current Answer: {current_answer}\n\n"
        "Enhanced Answer:"
    )
    return call_gemini(prompt)

# ═══════════════════════════ KB MUTATION HELPERS ══════════════════════

def upsert_qa(kb: dict, question: str, answer: str, confidence: float = 1.0):
    """
    Insert or update a Q&A pair in the knowledge base.
    Merges duplicate questions (case-insensitive) instead of creating duplicates.
    """
    now = datetime.now().isoformat()
    q_lower = question.lower().strip()

    for pair in kb["qa_pairs"]:
        if pair["question"].lower().strip() == q_lower:
            # Update existing entry
            pair["answer"]     = answer
            pair["confidence"] = confidence
            pair["updated_at"] = now
            save_kb(kb)
            return

    # Insert new entry
    new_id = max((p["id"] for p in kb["qa_pairs"]), default=0) + 1
    kb["qa_pairs"].append({
        "id":          new_id,
        "question":    question,
        "answer":      answer,
        "tags":        [],
        "confidence":  confidence,
        "updated_at":  now,
        "asked_count": 1
    })
    save_kb(kb)


def store_pending(kb: dict, question: str):
    """
    Save an unanswered question to the pending store.
    Increments asked_count if already pending.
    """
    now     = datetime.now().isoformat()
    q_lower = question.lower().strip()

    for pq in kb["pending_questions"]:
        if pq["question"].lower().strip() == q_lower:
            pq["asked_count"] += 1
            save_kb(kb)
            return

    new_id = max((p["id"] for p in kb["pending_questions"]), default=0) + 1
    kb["pending_questions"].append({
        "id":          new_id,
        "question":    question,
        "answer":      None,
        "asked_count": 1,
        "first_asked": now,
        "status":      "pending"
    })
    save_kb(kb)

# ═══════════════════════════ AUTO-UPDATE THREAD ═══════════════════════

def auto_update_loop():
    """
    Background daemon thread that runs every UPDATE_INTERVAL seconds.
    Tasks:
      1. Resolve pending questions with fresh Gemini answers.
      2. Enhance existing answers that are short or low-confidence.
    """
    print(f"[Auto-update] Thread started — cycle every {UPDATE_INTERVAL}s")
    while True:
        time.sleep(UPDATE_INTERVAL)
        print(f"[Auto-update] Running update cycle at {datetime.now().isoformat()}")
        try:
            kb = load_kb()

            # ── Task 1: Resolve pending questions ───────────────────
            pending = [p for p in kb["pending_questions"] if p["status"] == "pending"]
            # Prioritise questions asked most often
            pending.sort(key=lambda x: x["asked_count"], reverse=True)

            for pq in pending[:MAX_PENDING_RESOLVE]:
                answer = generate_answer(pq["question"])
                if answer:
                    upsert_qa(kb, pq["question"], answer)
                    pq["answer"] = answer
                    pq["status"] = "resolved"
                    kb = load_kb()   # reload after upsert save
                    print(f"[Auto-update] Resolved: '{pq['question'][:60]}'")

            save_kb(kb)

            # ── Task 2: Enhance short / low-confidence answers ──────
            kb = load_kb()
            for pair in kb["qa_pairs"]:
                needs_enhancement = (
                    len(pair.get("answer", "")) < SHORT_ANSWER_LEN
                    or pair.get("confidence", 1.0) < 0.7
                )
                if needs_enhancement:
                    enhanced = enhance_answer(pair["question"], pair["answer"])
                    if enhanced:
                        pair["answer"]     = enhanced
                        pair["confidence"] = 1.0
                        pair["updated_at"] = datetime.now().isoformat()
                        print(f"[Auto-update] Enhanced: '{pair['question'][:60]}'")

            save_kb(kb)

        except Exception as exc:
            print(f"[Auto-update ERROR] {exc}")


# ═══════════════════════════ CORE CHAT HANDLER ════════════════════════

def process_message(user_message: str) -> dict:
    """
    Main chat logic:
      1. Search knowledge base for a similar question.
      2a. Found → return answer (enhance if short).
      2b. Not found → ask Gemini, store result, return answer.
      2c. Gemini fails → store as pending, return apology.
    """
    kb = load_kb()
    match, score = find_best_match(user_message, kb)

    # ── Branch A: knowledge base hit ────────────────────────────────
    if match:
        match["asked_count"] = match.get("asked_count", 0) + 1

        # Enhance if the answer is too short
        if len(match["answer"]) < SHORT_ANSWER_LEN or match.get("confidence", 1.0) < 0.7:
            enhanced = enhance_answer(match["question"], match["answer"])
            if enhanced:
                match["answer"]     = enhanced
                match["confidence"] = 1.0
                match["updated_at"] = datetime.now().isoformat()

        save_kb(kb)
        return {
            "response": match["answer"],
            "source":   "knowledge_base",
            "score":    round(score, 3),
            "question": match["question"]
        }

    # ── Branch B: knowledge base miss → ask Gemini ──────────────────
    answer = generate_answer(user_message)
    if answer:
        kb = load_kb()
        upsert_qa(kb, user_message, answer)
        return {
            "response": answer,
            "source":   "gemini_new",
            "score":    round(score, 3),
            "question": user_message
        }

    # ── Branch C: Gemini unavailable → store pending ─────────────────
    kb = load_kb()
    store_pending(kb, user_message)
    return {
        "response": (
            "Sorry, I don't have information on that right now. 😔\n"
            "I've saved your question and will update my knowledge base "
            "as soon as I find reliable information. Please ask again later! 🔄"
        ),
        "source":   "pending",
        "score":    0.0,
        "question": user_message
    }

# ═══════════════════════════ HTTP SERVER ══════════════════════════════

class ChatHandler(BaseHTTPRequestHandler):
    """Minimal single-file HTTP handler — no Flask required."""

    def log_message(self, fmt, *args):
        # Custom access log format
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {fmt % args}")

    # ── CORS headers ──────────────────────────────────────────────────
    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    # ── Send JSON response ────────────────────────────────────────────
    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._set_cors()
        self.end_headers()
        self.wfile.write(body)

    # ── Send HTML file ────────────────────────────────────────────────
    def _send_html(self, path: str):
        try:
            with open(path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._set_cors()
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    # ── GET routes ────────────────────────────────────────────────────
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_html("index.html")

        elif self.path == "/kb":
            # Expose full knowledge base (for dashboard / debugging)
            self._send_json(load_kb())

        elif self.path == "/stats":
            # Quick statistics endpoint
            kb = load_kb()
            self._send_json({
                "total_qa":      len(kb["qa_pairs"]),
                "pending":       len([p for p in kb["pending_questions"] if p["status"] == "pending"]),
                "resolved":      len([p for p in kb["pending_questions"] if p["status"] == "resolved"]),
                "last_updated":  kb["metadata"]["last_updated"],
                "bot_name":      kb["metadata"]["bot_name"],
                "author":        kb["metadata"]["author"]
            })
        else:
            self.send_response(404)
            self.end_headers()

    # ── POST routes ───────────────────────────────────────────────────
    def do_POST(self):
        if self.path == "/chat":
            length  = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            message = payload.get("message", "").strip()

            if not message:
                self._send_json({"error": "Empty message"}, status=400)
                return

            result = process_message(message)
            self._send_json(result)

        else:
            self.send_response(404)
            self.end_headers()

    # ── OPTIONS (preflight CORS) ───────────────────────────────────────
    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors()
        self.end_headers()


# ═══════════════════════════ ENTRY POINT ══════════════════════════════

if __name__ == "__main__":
    # ── Start background auto-update thread ───────────────────────────
    updater = threading.Thread(target=auto_update_loop, daemon=True)
    updater.start()

    # ── Start HTTP server ─────────────────────────────────────────────
    server = HTTPServer(("0.0.0.0", PORT), ChatHandler)

    print("=" * 52)
    print("   🤖  NEW GEN AI Chat Bot  |  by Anubhav")
    print("=" * 52)
    print(f"   🌐  http://localhost:{PORT}")
    print(f"   📚  Knowledge Base : {KB_FILE}")
    print(f"   🔄  Auto-update    : every {UPDATE_INTERVAL // 60} min")
    print("   Press Ctrl+C to stop")
    print("=" * 52)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down gracefully.")
        server.server_close()
