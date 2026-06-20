"""
Project 1: Rule-Based AI Chatbot
DecodeLabs Industrial Training Kit - Batch 2026

Goal:
    Create a simple rule-based chatbot that responds to predefined
    user inputs using pure if-else / dictionary-lookup logic (no ML).

Architecture (IPO Model):
    INPUT   -> Sanitization & Normalization (lower-casing, stripping)
    PROCESS -> Intent Matching using a dictionary (Knowledge Base)
    OUTPUT  -> Response Generation + Fallback for unknown inputs

Key Requirements covered:
    1. Continuous input loop (while True)
    2. Sanitization of raw input (case & whitespace handling)
    3. Knowledge base with 5+ intents (dictionary, O(1) lookup)
    4. Fallback response for unrecognized input
    5. Clean exit strategy (kill command -> break)
"""

import random
from datetime import datetime

# ---------------------------------------------------------------------------
# PHASE 1: KNOWLEDGE BASE (The Logic Skeleton)
# ---------------------------------------------------------------------------
# Using a dictionary instead of a long if-elif ladder gives us O(1) constant
# time lookup instead of O(n) linear scanning (see "Algorithmic Efficiency").
# Each key (intent) maps to a list of possible bot replies for variety.

knowledge_base = {
    "hello": ["Hi there! How can I help you today?", "Hello! Good to see you."],
    "hi": ["Hey! What can I do for you?", "Hi! How's it going?"],
    "how are you": ["I'm just a bunch of if-else statements, but I'm doing great!",
                     "Running smoothly, thanks for asking!"],
    "what is your name": ["I'm DecodeBot, your friendly rule-based assistant.",
                           "You can call me DecodeBot."],
    "help": ["I can chat about greetings, my name, the time, and how I'm doing. "
              "Try saying 'hello', 'time', or 'bye'."],
    "time": [],  # handled dynamically below
    "thank you": ["You're welcome!", "Anytime!"],
    "thanks": ["You're welcome!", "Happy to help!"],
}

# Commands that should end the continuous loop (the "Kill Command")
EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye"}

# Fallback response shown when no intent matches
FALLBACK_RESPONSE = "I do not understand that yet. Try 'help' to see what I can do."


# ---------------------------------------------------------------------------
# PHASE 2: SANITIZATION
# ---------------------------------------------------------------------------
def sanitize_input(raw_input: str) -> str:
    """Normalize raw user input: lower-case + strip leading/trailing spaces."""
    return raw_input.lower().strip()


# ---------------------------------------------------------------------------
# PHASE 3: INTENT MATCHING / RESPONSE GENERATION
# ---------------------------------------------------------------------------
def get_response(clean_input: str) -> str:
    """
    Look up the cleaned input in the knowledge base and return a reply.
    Uses the .get() method for an atomic lookup + fallback in one line.
    A couple of dynamic intents (like 'time') get special handling first.
    """
    # Dynamic / nested-condition intent: current time
    if "time" in clean_input:
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}."

    # Nested condition example: personalize greeting if a name is mentioned
    if clean_input.startswith("my name is"):
        name = clean_input.replace("my name is", "").strip().title()
        if name:
            return f"Nice to meet you, {name}!"

    # Standard dictionary lookup with fallback (Knowledge Base)
    options = knowledge_base.get(clean_input)
    if options:
        return random.choice(options)

    return FALLBACK_RESPONSE


# ---------------------------------------------------------------------------
# PHASE 4: THE HEARTBEAT (Continuous Loop)
# ---------------------------------------------------------------------------
def run_chatbot():
    """Main driver: runs the chatbot until an exit command is given."""
    print("=" * 50)
    print(" DecodeBot - Rule-Based AI Chatbot (Project 1)")
    print(" Type 'help' for options, or 'bye'/'exit' to quit.")
    print("=" * 50)

    while True:
        raw_input_text = input("You: ")
        clean_input = sanitize_input(raw_input_text)

        # Exit strategy: clean break command
        if clean_input in EXIT_COMMANDS:
            print("Bot: Goodbye! Have a great day. 👋")
            break

        # Guard against empty input
        if not clean_input:
            print(f"Bot: {FALLBACK_RESPONSE}")
            continue

        reply = get_response(clean_input)
        print(f"Bot: {reply}")


if __name__ == "__main__":
    run_chatbot()
