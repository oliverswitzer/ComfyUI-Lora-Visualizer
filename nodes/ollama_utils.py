"""
Shared utilities for interacting with the Ollama API
---------------------------------------------------

This module centralizes common functionality used by multiple nodes
when talking to the Ollama chat API.  In particular it provides
helpers to ensure a model is available locally (downloading it if
necessary) and to send chat messages to the model.  By factoring
these routines into a single place we avoid repeating the same
request and error handling logic across the PromptSplitter and
LoRAPromptComposer nodes, making future maintenance simpler.

The helpers intentionally accept an optional ``requests_module``
parameter so that unit tests can inject a mock implementation of
``requests`` without having to monkeypatch the global import in this
module.  Similarly, a ``status_channel`` can be provided to send
progress updates to the ComfyUI frontend via ``PromptServer``.

Note: This file should not import any node classes to avoid
circular dependencies.  It may, however, import ``PromptServer``
from the ComfyUI server if available for sending messages.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

try:
    import requests  # type: ignore[import]
except Exception:
    requests = None  # requests may not be available in some environments

try:
    from server import PromptServer  # type: ignore
except Exception:
    PromptServer = None  # PromptServer may not be available during tests


def ensure_model_available(
    model_name: str,
    api_url: str,
    *,
    requests_module: Any = None,
    status_channel: Optional[str] = None,
) -> None:
    """Ensure a model is available locally by checking and optionally pulling.

    This helper queries the ``/api/tags`` endpoint to determine
    whether the specified model is installed.  If not, it attempts to
    download the model via the ``/api/pull`` endpoint.  Status
    messages will be sent to the ComfyUI frontend on the provided
    ``status_channel`` (if both the channel and ``PromptServer`` are
    available).  Any exceptions raised during HTTP requests are
    caught and printed to the console, but otherwise ignored.

    Parameters:
        model_name: The name of the model to check or download.
        api_url: The full URL of the Ollama chat endpoint
            (e.g. ``http://localhost:11434/api/chat``).  The helper
            derives the base URL by removing ``/api/chat`` if present.
        requests_module: Optional substitute for the ``requests``
            library.  If ``None``, falls back to the globally
            imported ``requests`` object.  If no ``requests``
            implementation is available, the function returns
            silently.
        status_channel: Optional name of the message channel to send
            progress updates to via ``PromptServer.instance.send_sync``.

    Returns:
        None.  On failure, error messages are printed but the
        exception is swallowed to avoid crashing the caller.
    """
    # Determine which requests implementation to use
    req = requests_module or requests
    if req is None or not api_url:
        return
    base = api_url.rstrip("/")
    if base.endswith("/api/chat"):
        base = base[: -len("/api/chat")]
    tags_url = f"{base}/api/tags"
    pull_url = f"{base}/api/pull"
    # Check installed models
    try:
        resp = req.get(tags_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        installed: List[str] = [m.get("name") for m in data.get("models", [])]
        if model_name in installed:
            return
    except Exception as e:
        print(f"ollama_utils: failed to query installed models: {e}")
        return
    # Inform user about download
    msg = f"Ollama model '{model_name}' not found. Downloading..."
    if status_channel and PromptServer is not None:
        try:
            PromptServer.instance.send_sync(status_channel, {"status": msg})
        except Exception:
            pass
    else:
        print(msg)
    # Download model
    try:
        payload = {"model": model_name, "stream": False}
        resp = req.post(pull_url, json=payload, timeout=300)
        resp.raise_for_status()
    except Exception as e:
        print(f"ollama_utils: failed to download model '{model_name}': {e}")
        return
    done_msg = f"Model '{model_name}' downloaded successfully."
    if status_channel and PromptServer is not None:
        try:
            PromptServer.instance.send_sync(status_channel, {"status": done_msg})
        except Exception:
            pass
    else:
        print(done_msg)


def call_ollama_chat(
    system_prompt: str,
    user_content: str,
    *,
    model_name: str,
    api_url: str,
    timeout: int = 60,
    requests_module: Any = None,
) -> str:
    """Send a chat completion request to an Ollama model and return the reply.

    Constructs a chat payload with the given system and user messages,
    posts it to the configured API URL, and returns the assistant's
    content string.  This helper does not attempt to parse the
    returned content as JSON; it simply returns the raw string,
    leaving further processing to the caller.  If any error occurs
    (e.g. network failures, JSON parsing errors in the API response),
    an empty string is returned.

    Parameters:
        system_prompt: Text to send as the system message.
        user_content: Text to send as the user message.
        model_name: Name of the model to query.
        api_url: URL of the Ollama chat endpoint.
        timeout: Number of seconds to wait for the response.
        requests_module: Optional ``requests`` replacement.

    Returns:
        The assistant's reply content as a string, or an empty
        string on error.
    """
    req = requests_module or requests
    if req is None:
        print("ollama_utils: 'requests' library not available; cannot contact Ollama.")
        return ""

    # Construct chat messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Prepare the payload
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
    }

    try:
        resp = req.post(api_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        # Extract assistant content depending on API version
        if isinstance(data, dict):
            if "message" in data:
                content = data["message"].get("content", "")
            elif "choices" in data and data["choices"]:
                content = data["choices"][0].get("message", {}).get("content", "")
            else:
                content = ""
        else:
            content = ""
        return str(content).strip()
    except Exception as e:
        print(f"ollama_utils: error contacting Ollama: {e}")
        return ""


def send_chat(
    model_name: str,
    api_url: str,
    messages: List[Dict[str, Any]],
    timeout: int = 60,
    *,
    requests_module: Any = None,
) -> str:
    """Compatibility wrapper to send a chat completion request via Ollama.

    Many existing nodes expect a helper named ``send_chat`` that accepts
    a list of messages (with roles ``system`` and ``user``) and
    returns the assistant's response as a plain string.  This
    wrapper unpacks the system and user messages from the list and
    delegates to :func:`call_ollama_chat`.

    Parameters:
        model_name: Name of the Ollama model to query.
        api_url: URL of the Ollama chat endpoint.
        messages: A list of dictionaries with at least two entries: the
            first should have ``role`` equal to ``"system"`` and
            contain the system prompt under ``"content"``; the second
            should have ``role`` equal to ``"user"`` and contain the
            user content.  Any additional messages are ignored.
        timeout: Number of seconds to wait for the response.
        requests_module: Optional replacement for the ``requests``
            library.  If provided, it will be passed to
            :func:`call_ollama_chat`.

    Returns:
        The assistant's reply content as a string, or an empty
        string on error.
    """
    # Validate and unpack messages
    system_prompt = ""
    user_content = ""
    if isinstance(messages, list):
        # Find system and user messages in order
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "system" and not system_prompt:
                system_prompt = str(msg.get("content", ""))
            elif role == "user" and not user_content:
                user_content = str(msg.get("content", ""))
            # Break early if both have been found
            if system_prompt and user_content:
                break
    # Delegate to call_ollama_chat
    return call_ollama_chat(
        system_prompt,
        user_content,
        model_name=model_name,
        api_url=api_url,
        timeout=timeout,
        requests_module=requests_module,
    )
