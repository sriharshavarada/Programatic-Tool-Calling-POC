"""CLI wrapper: Traditional no-programmatic-tool-calling demo."""

from __future__ import annotations

import json

from demo_engine import run_traditional


def main() -> None:
    result = run_traditional()
    for item in result.logs:
        print(item.message)
    print("[TRADITIONAL] ===== SUMMARY =====")
    print(f"[TRADITIONAL] ok={result.ok} duration_ms={result.duration_ms}")
    print(f"[TRADITIONAL] token_usage={json.dumps(result.token_usage, default=str)}")
    print(f"[TRADITIONAL] metadata={json.dumps(result.metadata, default=str)}")
    if result.final_text:
        print("[TRADITIONAL] Final text:")
        print(result.final_text)
    if result.error:
        print(f"[TRADITIONAL] ERROR: {result.error}")


if __name__ == "__main__":
    main()
