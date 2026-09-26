"""Stage 0.1: send one question to the LLM and print the answer.

Concept: an LLM API call is just an HTTP request. You send a list of messages
and a model name; you get back the model's reply.

Run: python stage0/01_hello_openai.py
"""

import os
import sys

import openai
from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY and OPENAI_MODEL from .env

api_key = os.getenv("OPENAI_API_KEY", "")
model = os.getenv("OPENAI_MODEL") or "gpt-5.6-luna"  # default when .env leaves it empty
if not api_key or api_key == "paste-your-key-here":
    sys.exit("OPENAI_API_KEY is missing. Paste your key into .env as OPENAI_API_KEY=sk-...")

client = openai.OpenAI(api_key=api_key)
try:
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": "In two sentences, what is operating margin and why do consultants care about it?",
            }
        ],
    )
except openai.AuthenticationError:
    sys.exit("The API key was rejected. Check OPENAI_API_KEY in .env (no quotes or spaces).")
except openai.NotFoundError:
    sys.exit(f"Model '{model}' was not found. Check OPENAI_MODEL in .env against your account's model list.")
except openai.PermissionDeniedError as exc:
    sys.exit(f"Your account cannot use '{model}': {exc.message}")
except openai.RateLimitError as exc:
    sys.exit(f"Rate limit or quota problem (often: no credits on the account): {exc.message}")
except openai.APIConnectionError:
    sys.exit("Could not reach the OpenAI API. Check your internet connection.")

print(response.choices[0].message.content)
print(
    f"\n[model: {response.model}, tokens in: {response.usage.prompt_tokens}, out: {response.usage.completion_tokens}]"
)
