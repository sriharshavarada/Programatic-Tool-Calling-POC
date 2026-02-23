"""CLI wrapper: LangGraph sandbox/programmatic tool calling demo."""

from __future__ import annotations

import json

from demo_engine import run_sandbox_langgraph


def main() -> None:
    result = run_sandbox_langgraph()
    for item in result.logs:
        print(item.message)
    print("[SANDBOX] ===== SUMMARY =====")
    print(f"[SANDBOX] ok={result.ok} duration_ms={result.duration_ms}")
    print(f"[SANDBOX] token_usage={json.dumps(result.token_usage, default=str)}")
    print(f"[SANDBOX] metadata={json.dumps(result.metadata, default=str)}")
    if result.generated_code:
        print("[SANDBOX] ----- REMOTE GENERATED CODE START -----")
        print(result.generated_code)
        print("[SANDBOX] ----- REMOTE GENERATED CODE END -----")
    if result.final_text:
        print("[SANDBOX] Final text:")
        print(result.final_text)
    if result.error:
        print(f"[SANDBOX] ERROR: {result.error}")


if __name__ == "__main__":
    main()
