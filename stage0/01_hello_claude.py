"""Stage 0.1: send one question to Claude and print the answer.

Concept: an LLM API call is just an HTTP request. You send a list of messages
and a model name; you get back a list of content blocks.

Run: python stage0/01_hello_claude.py
"""

import anthropic
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from .env into the environment

MODEL = "claude-haiku-4-5"  # cheap and fast; used for all development calls

client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY automatically

response = client.messages.create(
    model=MODEL,
    max_tokens=500,
    messages=[
        {"role": "user", "content": "In two sentences, what is operating margin and why do consultants care about it?"}
    ],
)

for block in response.content:
    if block.type == "text":
        print(block.text)

print(f"\n[tokens in: {response.usage.input_tokens}, out: {response.usage.output_tokens}]")
