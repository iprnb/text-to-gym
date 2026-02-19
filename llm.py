"""
LLM provider wrappers: Ollama Cloud and OpenAI.

All functions return a plain string – either the model's response or an
"Error: …" message so callers can treat them uniformly.
"""

import json
import traceback
from typing import Optional

from config import MODEL_PROVIDER, OLLAMA_API_KEY, OLLAMA_MODEL, OPENAI_API_KEY, OPENAI_MODEL


# ---------------------------------------------------------------------------
# Provider-specific callers
# ---------------------------------------------------------------------------

def call_ollama(system_prompt: str, user_prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call the Ollama Cloud API and return the response text."""
    try:
        from ollama import Client

        if not OLLAMA_API_KEY:
            return "Error: OLLAMA_API_KEY not set. Export it as an environment variable."

        print(f"\n{'='*60}\nOLLAMA CLOUD API CALL\nModel: {model}\n{'='*60}\n")

        client = Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
            timeout=120
        )
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.2, "num_predict": 10_000, "stop": None},
        )

        content = response.message.content
        print(f"Response received! Length: {len(content)} chars")
        print(f"Preview: {content[:200]}...")

        if not content:
            return "Error: Ollama Cloud returned empty content."
        return content

    except AttributeError as e:
        return f"Error: Could not access response content. {e}"
    except Exception as e:
        return f"Error calling Ollama Cloud API: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def call_openai(system_prompt: str, user_prompt: str, model: str = OPENAI_MODEL) -> str:
    """Call the OpenAI API and return the response text."""
    try:
        from openai import OpenAI

        if not OPENAI_API_KEY:
            return "Error: OPENAI_API_KEY not set."

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=3_000,
        )
        return response.choices[0].message.content

    except ImportError:
        return "Error: openai package not installed. Run: pip install openai"
    except Exception as e:
        return f"Error calling OpenAI API: {e}"


# ---------------------------------------------------------------------------
# Unified caller
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str, provider: str = MODEL_PROVIDER) -> str:
    """Dispatch to the configured LLM provider."""
    if provider == "ollama":
        return call_ollama(system_prompt, user_prompt)
    if provider == "openai":
        return call_openai(system_prompt, user_prompt)
    return f"Error: Unknown provider '{provider}'."


# ---------------------------------------------------------------------------
# JSON response parser
# ---------------------------------------------------------------------------

def parse_llm_response(response: str) -> Optional[dict]:
    """
    Parse a JSON response from the LLM.

    Tries strict JSON first, then falls back to extracting the first
    ``{…}`` block in case the model added surrounding text.
    Returns ``None`` on failure.
    """
    if not response:
        return None
    if response.startswith("Error:"):
        print(f"LLM Error: {response}")
        return None

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(response[start:end])
    except Exception as e:
        print(f"JSON parsing error: {e}")
        print(f"Response preview: {response[:500]}...")

    return None
