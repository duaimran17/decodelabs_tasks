# Project 1: Rule-Based AI Chatbot 🤖

**Track:** DecodeLabs Industrial Training Kit — Batch 2026
**Module:** Foundation Phase — Control Flow & Logic

## 📌 Goal
Create a simple rule-based chatbot that responds to predefined user inputs
using explicit logic (dictionary lookup), not machine learning.

## ✅ Requirements Covered
| Requirement | Implementation |
|---|---|
| Continuous loop | `while True` loop in `run_chatbot()` |
| Sanitization | `.lower().strip()` on every input |
| Knowledge base (5+ intents) | Python dictionary with 8 intents |
| Fallback response | `FALLBACK_RESPONSE` returned for unknown input |
| Exit strategy | `EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye"}` triggers `break` |
| If-else / decision logic | Nested conditions for dynamic intents (time, name) |

## 🧠 How It Works (IPO Model)
1. **Input** – Raw text is captured via `input()`.
2. **Process** – Text is sanitized, then matched against the knowledge base
   dictionary (O(1) lookup) instead of a long if-elif ladder (O(n)).
3. **Output** – A response is returned, either from the dictionary, a
   dynamic handler (like the current time), or the fallback message.

## ▶️ How to Run
```bash
python3 chatbot.py
```

## 💬 Example Session
```
You: hello
Bot: Hi there! How can I help you today?
You: my name is Ali
Bot: Nice to meet you, Ali!
You: time
Bot: The current time is 10:30:03.
You: bye
Bot: Goodbye! Have a great day. 👋
```

## 🛠 Possible Extensions
- Add more intents to `knowledge_base`
- Add a personality / tone setting
- Swap the dictionary for a hybrid: rule match → else pass to an LLM
