# 🤖 NEW GEN AI Chat Bot

> **Author:** Anubhav &nbsp;|&nbsp; **Engine:** Google Gemini 1.5 Flash &nbsp;|&nbsp; **Storage:** JSON Knowledge Base

A **dynamically self-expanding AI chatbot** that automatically learns from every conversation, enhances weak answers, and resolves unanswered questions over time — with zero external database or web framework.

---

## 📋 Problem Statement

Standard chatbots have a fixed knowledge base. When users ask new questions the bot hasn't seen, it simply fails. This project solves that by building a chatbot that:

1. Searches a local JSON knowledge base using **TF-IDF cosine similarity**
2. Falls back to **Google Gemini** for new questions — and **stores the answer permanently**
3. Automatically **enhances short/incomplete answers** in the background
4. **Stores unanswered questions** as pending and resolves them on the next auto-update cycle

The knowledge base grows with every interaction. The bot gets smarter over time.

---

## 🗂 Project Structure

```
new-gen-ai-chatbot/
├── chatbot.py           ← Python server (http.server + Gemini + auto-update)
├── index.html           ← Simple chat UI (no framework)
├── knowledge_base.json  ← Q&A store + pending questions
├── NEW_GEN_AI_ChatBot.ipynb  ← Google Colab notebook (public URL via ngrok)
└── README.md
```

**Only 4 source files. Zero external web framework.**

---

## ⚙️ Methodology

### 1. Vector Similarity Search (TF-IDF)
Questions in the knowledge base are compared to the user's query using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization with bigrams, followed by **cosine similarity**. If the highest similarity score exceeds `SIMILARITY_TH = 0.40`, the stored answer is returned.

```
User Query ──► TF-IDF Vectorizer ──► Cosine Similarity ──► Best Match
                                                            ↓
                                               score ≥ 0.40 ? KB answer : Gemini
```

### 2. Dynamic Knowledge Expansion
| Scenario | Action |
|---|---|
| Query matches KB (score ≥ 0.40) | Return stored answer; enhance if short |
| Query doesn't match KB | Call Gemini → store new Q&A pair |
| Gemini unavailable | Store as `pending`; resolve on next cycle |

### 3. Answer Enhancement
If an existing answer is shorter than 180 characters or has `confidence < 0.7`, Gemini is called to expand it with bullet points and examples. The KB entry is updated in-place.

### 4. Auto-Update Background Thread
A daemon thread runs every **30 minutes** and:
- Resolves up to 5 pending questions (prioritized by `asked_count`)
- Enhances short answers in the main KB

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install google-generativeai scikit-learn numpy pyngrok
```

### 1. Get a Gemini API Key
Visit [aistudio.google.com](https://aistudio.google.com) → Get API Key → Copy it

### 2. Set your API key
```bash
export GEMINI_API_KEY="your-key-here"
# OR edit chatbot.py line 22: GEMINI_API_KEY = "your-key-here"
```

### 3. Run the server
```bash
python chatbot.py
```

Open `http://localhost:8080` in your browser. That's it!

### 4. Google Colab (recommended for sharing)
Open `NEW_GEN_AI_ChatBot.ipynb` in Colab → Run all cells → Get a public URL via ngrok

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/new-gen-ai-chatbot/blob/main/NEW_GEN_AI_ChatBot.ipynb)

---

## 📊 Knowledge Base Schema

### `qa_pairs` — Main Answer Store
```json
{
  "id": 1,
  "question": "What is artificial intelligence?",
  "answer": "AI simulates human intelligence in machines...",
  "tags": ["AI", "basics"],
  "confidence": 1.0,
  "updated_at": "2025-01-01T12:00:00",
  "asked_count": 5
}
```

### `pending_questions` — Unanswered Queue
```json
{
  "id": 1,
  "question": "What is quantum computing?",
  "answer": null,
  "asked_count": 3,
  "first_asked": "2025-01-01T12:00:00",
  "status": "pending"
}
```

When Gemini provides an answer for a pending question, `status` changes to `"resolved"` and the Q&A pair is moved to `qa_pairs`.

---

## 🌐 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `POST` | `/chat` | Send message, get response |
| `GET` | `/kb` | Full knowledge base JSON |
| `GET` | `/stats` | Quick stats (total QA, pending, resolved) |

### Example
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is machine learning?"}'
```
```json
{
  "response": "Machine Learning is a subset of AI...",
  "source": "knowledge_base",
  "score": 0.847
}
```

**`source` values:**
- `knowledge_base` — answered from stored Q&A
- `gemini_new` — new answer from Gemini (now stored)
- `pending` — couldn't answer; saved for later

---

## 🔧 Configuration

Edit these constants at the top of `chatbot.py`:

| Variable | Default | Description |
|---|---|---|
| `SIMILARITY_TH` | `0.40` | Min cosine score to count as a KB match |
| `SHORT_ANSWER_LEN` | `180` | Answers shorter than this get enhanced |
| `UPDATE_INTERVAL` | `1800` | Auto-update cycle in seconds (30 min) |
| `MAX_PENDING_RESOLVE` | `5` | Max pending Qs resolved per cycle |
| `PORT` | `8080` | HTTP server port |

---

## 📈 Visualizations (Colab Notebook)

The notebook generates:

1. **Answer Length Distribution** — histogram of KB answer lengths
2. **Tag Frequency Bar Chart** — most common topics in the KB
3. **Confidence Score Pie Chart** — proportion of high/low confidence answers
4. **Simulated KB Growth** — projected growth over 30 days
5. **Question Similarity Heatmap** — TF-IDF cosine similarity between all stored questions

---

## 🔄 How the Bot Self-Improves

```
User asks: "Explain transformers in NLP"
                  │
         ┌────────▼────────┐
         │  TF-IDF Search  │  score = 0.21 < 0.40 → no match
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  Gemini API     │  generates comprehensive answer
         └────────┬────────┘
                  │
         ┌────────▼──────────────┐
         │  upsert_qa() in JSON  │  stored permanently
         └───────────────────────┘

Next time someone asks "What are transformers?":
  TF-IDF score = 0.73 → returns stored answer instantly ⚡
```

---

## 🛠 Dependencies

| Package | Purpose |
|---|---|
| `google-generativeai` | Gemini 1.5 Flash API |
| `scikit-learn` | TF-IDF vectorizer + cosine similarity |
| `numpy` | Array operations |
| `pyngrok` | Public URL tunneling (Colab) |

Standard library: `json`, `http.server`, `threading`, `time`, `datetime`, `os`

---

## 🔮 Future Improvements

- [ ] Replace TF-IDF with Gemini embeddings for semantic similarity
- [ ] Add a web scraping module to auto-seed knowledge from URLs
- [ ] Implement conversation history (multi-turn context)
- [ ] Add admin dashboard to manually edit/delete KB entries
- [ ] Export KB to SQLite or ChromaDB for production scale

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ by **Anubhav** · Powered by Google Gemini 1.5 Flash*
