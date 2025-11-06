#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ELIZA-like chatbot with pluggable rule sets.
Default domain: IT Helpdesk triage (different from Rogerian psychotherapist).

How it works (high level):
- Incoming text is normalized and lightly tokenized.
- We apply ordered regex rules. The first match wins.
- Rules can include capture groups; captured text is passed through a
  reflection map (I->you, my->your, etc.) and substituted into response templates.
- If no rule matches, we fall back to generic prompts.

Run:
    python ELIZA.py

Type 'quit', 'exit', or 'bye' to end the session.
"""

from __future__ import annotations
import re
import random
import sys
from typing import Callable, Dict, List, Pattern, Tuple

# -----------------------------
# Reflection map (pronoun swap)
# -----------------------------
REFLECTIONS: Dict[str, str] = {
    "i": "you",
    "i'm": "you're",
    "im": "you're",
    "i am": "you are",
    "i'd": "you'd",
    "i've": "you've",
    "i'll": "you'll",
    "my": "your",
    "me": "you",
    "mine": "yours",
    "myself": "yourself",
    "you": "I",
    "you're": "I'm",
    "you are": "I am",
    "you'd": "I'd",
    "you've": "I've",
    "you'll": "I'll",
    "your": "my",
    "yours": "mine",
    "yourself": "myself",
}

# Helper to reflect a phrase (very simple, token-based)
_word_re = re.compile(r"\b[\w']+\b")


def reflect(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        w = m.group(0)
        lw = w.lower()
        return REFLECTIONS.get(lw, w)

    return _word_re.sub(repl, text)


# -----------------------------
# Utility: simple normalization
# -----------------------------
_punct_space = re.compile(r"[\t\r\f\v]+")
_multi_space = re.compile(r"\s{2,}")


def normalize(s: str) -> str:
    s = s.strip()
    s = _punct_space.sub(" ", s)
    s = _multi_space.sub(" ", s)
    return s


# ---------------------------------
# Rule structure and rule evaluation
# ---------------------------------
class Rule:
    def __init__(self, pattern: str, responses: List[str], flags: int = re.IGNORECASE):
        # Use \A and \Z to anchor to full string if you want, but we keep flexible matching
        self.pattern: Pattern[str] = re.compile(pattern, flags)
        self.responses = responses

    def try_apply(self, text: str) -> str | None:
        m = self.pattern.search(text)
        if not m:
            return None
        # Build a response by substituting captures with reflected text
        template = random.choice(self.responses)

        # Replace placeholders {1}, {2}, ... with reflected groups
        def sub_group(match: re.Match[str]) -> str:
            idx = int(match.group(1))
            if idx <= m.lastindex:
                grp = m.group(idx)
                return reflect(grp)
            return ""

        return re.sub(r"\{(\d+)\}", sub_group, template)


# ---------------------------------
# Domain: IT Helpdesk triage rules
# ---------------------------------
HELPDESK_RULES: List[Rule] = [
    # Greetings / openers
    Rule(r"\b(hi|hello|hey|good\s*(morning|afternoon|evening))\b", [
        "Hi! What issue are you running into today?",
        "Hello—tell me what you're trying to do and what's going wrong.",
    ]),

    # Can't / cannot
    Rule(r"\b(i\s+can't|i\s+cannot|can't)\s+(.*)", [
        "What happens when you try to {2}? Any error messages?",
        "How long have you been unable to {2}? Did anything change recently?",
    ]),

    # Error codes/messages
    Rule(r"\berror\b\s*(code\s*)?(\w+)", [
        "Thanks—what action leads to error {2}?",
        "Got it. Is error {2} reproducible or intermittent?",
    ]),

    # Slow / performance
    Rule(r"\b(slow|lag(gy)?|freez(e|ing)|hang(s)?)\b", [
        "When did it start feeling {1}? Does it improve after a reboot?",
        "Noted it's {1}. Is CPU or memory unusually high when this occurs?",
    ]),

    # Network
    Rule(r"\b(wifi|wi-?fi|internet|network|vpn)\b.*\b(down|drop|disconnect|can't|cannot|fail|failing)\b", [
        "Network issue: are other sites/services affected, or just one?",
        "VPN/network problem noted—does it work on another network or hotspot?",
    ]),

    # Printer
    Rule(r"\b(print(er)?|printing)\b.*\b(issue|problem|won't|cant|can't|fail|error)\b", [
        "Printer issue: is it on the same network and showing online?",
        "Let’s check: correct printer selected, paper/ink OK, and any queue errors?",
    ]),

    # Credentials / login
    Rule(r"\b(log\s*in|login|sign\s*in|password)\b", [
        "Are you seeing a specific login error? When did you last change your password?",
        "Let's try: reset the password, confirm MFA works, then attempt sign-in again.",
    ]),

    # File access / permissions
    Rule(r"\b(access|permission|forbidden|denied|readonly|read-only)\b", [
        "Which file or folder shows that? Do you know the exact path or link?",
        "We may need to adjust permissions; who else should have access besides you?",
    ]),

    # Software install/update
    Rule(r"\b(install|update|upgrade|patch)\b", [
        "What are you trying to {1}? Do you see an admin prompt or restriction?",
        "Understood. Are you on the latest OS version and do you have install rights?",
    ]),

    # Device descriptions
    Rule(r"\b(mac|windows|linux|iphone|android|ipad|laptop|desktop|pc)\b", [
        "Thanks—what's the OS version on your {0}?",
        "Noted you're on {0}. Is the device managed by MDM/IT?",
    ]),

    # Catch-all with captured tail
    Rule(r"\b(i\s+(need|want)\s+help\s+with)\s+(.*)", [
        "Happy to help with {3}. What have you tried so far?",
        "Let’s focus on {3}. What changed just before this started?",
    ]),

    # Generic "you" statements -> reflect
    Rule(r"\byou\b\s+(are|were|seem|sound|made)\s+(.*)", [
        "What makes you think I {1} {2}?",
        "Why do you say I {1} {2}?",
    ]),

    # Generic "I am" -> reflect
    Rule(r"\bi\s+am\s+(.*)", [
        "How long have you been {1}?",
        "What do you think is causing you to be {1}?",
    ]),

    # Fallback mirrors a fragment
    Rule(r".*", [
        "Tell me more about that.",
        "Could you describe that in a bit more detail?",
        "What happened right before this?",
        "When did you first notice this?",
    ]),
]

# If you want the classic Rogerian set, you could define a ROGERIAN_RULES list
# and swap which list is used below.

# -----------------------------
# Chat loop
# -----------------------------
QUIT = {"quit", "exit", "bye", "goodbye", "q"}

PROMPTS = [
    "Hello! I'm your IT helpdesk assistant. What's going on?",
    "Hi there—what problem can I help you with today?",
]


def respond(user: str, rules: List[Rule]) -> str:
    text = normalize(user)
    for rule in rules:
        out = rule.try_apply(text)
        if out is not None:
            return normalize(out)
    # Should never hit due to catch-all, but just in case
    return random.choice([
        "I'm not sure I followed—could you rephrase?",
        "Can you give me a concrete example?",
    ])


def main() -> None:
    print(random.choice(PROMPTS))
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user:
            continue
        if user.lower() in QUIT:
            print("Goodbye!")
            break
        print(respond(user, HELPDESK_RULES))


if __name__ == "__main__":
    main()
