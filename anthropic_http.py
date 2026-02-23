"""Minimal Anthropic Messages API helper using Python stdlib only."""

from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicAPIError(RuntimeError):
    """Raised when the Anthropic API returns an error response."""


_DOTENV_LOADED = False


def load_dotenv_if_present(force: bool = False) -> None:
    """Load simple KEY=VALUE pairs from .env into os.environ (no override)."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED and not force:
        return

    candidate_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
    ]
    seen: set[Path] = set()

    for dotenv_path in candidate_paths:
        if dotenv_path in seen:
            continue
        seen.add(dotenv_path)
        if not dotenv_path.exists():
            continue
        try:
            text = dotenv_path.read_text(encoding="utf-8-sig")
        except OSError:
            continue

        loaded_any = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue
            if (
                len(value) >= 2
                and ((value[0] == value[-1]) and value[0] in {"'", '"'})
            ):
                value = value[1:-1]
            os.environ[key] = value
            loaded_any = True
        if loaded_any:
            _DOTENV_LOADED = True
            return

    # Mark as attempted to avoid repeated filesystem scans when no .env is present.
    _DOTENV_LOADED = True


def require_api_key() -> str:
    load_dotenv_if_present()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Retry once in case .env was added/edited after the process started.
        load_dotenv_if_present(force=True)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise AnthropicAPIError(
            "Missing ANTHROPIC_API_KEY. Set it in your shell before running this demo."
        )
    return api_key


def post_messages(payload: dict, log_prefix: str) -> dict:
    """POST to Anthropic Messages API and return parsed JSON."""
    api_key = require_api_key()
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    beta = os.environ.get("ANTHROPIC_BETA")
    if beta:
        headers["anthropic-beta"] = beta

    print(f"{log_prefix} HTTP POST {API_URL}")
    print(
        f"{log_prefix} Request summary: model={payload.get('model')!r}, "
        f"max_tokens={payload.get('max_tokens')}, "
        f"messages={len(payload.get('messages', []))}, "
        f"tools={len(payload.get('tools', []))}"
    )
    if "container" in payload:
        print(f"{log_prefix} Reusing container={payload['container']!r}")

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"{log_prefix} HTTP error status={exc.code}")
        print(f"{log_prefix} HTTP error body={body}")
        raise AnthropicAPIError(
            "Anthropic API request failed. If you see 'missing_beta_header', set "
            "ANTHROPIC_BETA=advanced-tool-use-2025-11-20 and retry."
        ) from exc
    except urllib.error.URLError as exc:
        raise AnthropicAPIError(f"Network error calling Anthropic API: {exc}") from exc


def extract_text_blocks(response: dict) -> list[str]:
    """Extract text blocks from a Messages API response."""
    texts: list[str] = []
    for block in response.get("content", []):
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
    return texts


def strip_code_fences(text: str) -> str:
    """Best-effort removal of markdown code fences if model adds them."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped
